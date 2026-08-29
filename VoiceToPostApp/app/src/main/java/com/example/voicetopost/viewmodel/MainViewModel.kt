package com.example.voicetopost.viewmodel

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Base64
import androidx.core.content.FileProvider
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.voicetopost.api.ConfirmPostRequest
import com.example.voicetopost.api.GenerateImageOption
import com.example.voicetopost.api.PostVariation
import com.example.voicetopost.api.RetrofitClient
import com.example.voicetopost.audio.AudioRecorder
import com.voicetopost.data.UserPrefs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.SocketTimeoutException
import java.net.URL

// ── UI States ─────────────────────────────────────────────────

sealed class RecordingState {
    object Idle       : RecordingState()
    object Recording  : RecordingState()
    object Processing : RecordingState()
    data class Error(val message: String) : RecordingState()
}

sealed class PostsState {
    object Empty   : PostsState()
    object Loading : PostsState()
    data class Success(val posts: List<PostVariation>, val isPartial: Boolean = false) : PostsState()
    data class Error(val message: String) : PostsState()
}

sealed class PublishState {
    object Idle        : PublishState()
    object Publishing  : PublishState()
    data class Success(val message: String) : PublishState()
    data class Error(val message: String)   : PublishState()
}

// State of image generation for a single post index
sealed class ImageGenState {
    object Idle    : ImageGenState()
    object Loading : ImageGenState()
    data class Success(val options: List<GenerateImageOption>) : ImageGenState()
    data class Error(val message: String) : ImageGenState()
}

// One-shot feedback for save-to-gallery / share actions
sealed class MediaActionState {
    object Idle    : MediaActionState()
    object Working : MediaActionState()
    data class Success(val message: String) : MediaActionState()
    data class Error(val message: String) : MediaActionState()
}

// ── ViewModel ─────────────────────────────────────────────────

class MainViewModel : ViewModel() {

    val platforms = listOf("linkedin", "twitter", "instagram")
    val tones     = listOf("professional", "casual", "witty", "inspirational", "informative")

    private val _platform = MutableStateFlow("linkedin")
    val platform: StateFlow<String> = _platform

    private val _tone = MutableStateFlow("professional")
    val tone: StateFlow<String> = _tone

    private val _recordingState = MutableStateFlow<RecordingState>(RecordingState.Idle)
    val recordingState: StateFlow<RecordingState> = _recordingState

    private val _postsState = MutableStateFlow<PostsState>(PostsState.Empty)
    val postsState: StateFlow<PostsState> = _postsState

    private val _publishState = MutableStateFlow<PublishState>(PublishState.Idle)
    val publishState: StateFlow<PublishState> = _publishState

    private val _recordingSeconds = MutableStateFlow(0)
    val recordingSeconds: StateFlow<Int> = _recordingSeconds

    private var recorder: AudioRecorder? = null
    private var currentAudioFile: File? = null

    private val _userId = MutableStateFlow("user_default")
    val userId: StateFlow<String> = _userId

    private val _connectedPlatforms = MutableStateFlow<Set<String>>(emptySet())
    val connectedPlatforms: StateFlow<Set<String>> = _connectedPlatforms

    // ── Image generation state (keyed by post index) ───────────

    private val _wantsImage = MutableStateFlow(false)
    val wantsImage: StateFlow<Boolean> = _wantsImage

    private val _imageGenStates = MutableStateFlow<Map<Int, ImageGenState>>(emptyMap())
    val imageGenStates: StateFlow<Map<Int, ImageGenState>> = _imageGenStates

    private val _selectedImages = MutableStateFlow<Map<Int, GenerateImageOption>>(emptyMap())
    val selectedImages: StateFlow<Map<Int, GenerateImageOption>> = _selectedImages

    private val _mediaActionState = MutableStateFlow<MediaActionState>(MediaActionState.Idle)
    val mediaActionState: StateFlow<MediaActionState> = _mediaActionState

    fun toggleWantsImage() { _wantsImage.value = !_wantsImage.value }

    fun resetMediaActionState() { _mediaActionState.value = MediaActionState.Idle }

    fun loadConnectedPlatforms(context: Context) {
        viewModelScope.launch {
            UserPrefs.getConnectedPlatforms(context).collect {
                _connectedPlatforms.value = it
            }
        }
    }

    fun markPlatformConnected(context: Context, platform: String) {
        viewModelScope.launch {
            UserPrefs.addConnectedPlatform(context, platform)
            _connectedPlatforms.update { it + platform }
        }
    }

    private var timerJob: Job? = null

    fun selectPlatform(p: String) { _platform.value = p }
    fun selectTone(t: String)     { _tone.value = t }
    fun resetPublishState()       { _publishState.value = PublishState.Idle }
    fun resetPosts()              { _postsState.value = PostsState.Empty }

    fun loadUserId(context: Context) {
        viewModelScope.launch {
            UserPrefs.getUserId(context).collect {
                _userId.value = it
            }
        }
    }

    fun onOAuthSuccess(context: Context, newUserId: String) {
        viewModelScope.launch {
            UserPrefs.saveOAuthUserId(context, newUserId)
            _userId.value = newUserId
        }
    }

    // ── Recording ─────────────────────────────────────────────

    fun startRecording(context: Context) {
        if (_recordingState.value is RecordingState.Recording) return
        try {
            recorder = AudioRecorder(context)
            currentAudioFile = recorder!!.start()
            _recordingState.value = RecordingState.Recording
            _recordingSeconds.value = 0
            startTimer()
        } catch (e: Exception) {
            _recordingState.value = RecordingState.Error("Cannot start recording: ${e.message}")
        }
    }

    fun stopAndGenerate() {
        if ((recorder?.recordingDurationMs ?: 0) < 2000) {
            _recordingState.value = RecordingState.Error(
                "Too short! Please speak for at least 2 seconds."
            )
            recorder?.cancel()
            recorder = null
            stopTimer()
            return
        }

        stopTimer()
        val file = recorder?.stop()
        recorder = null

        if (file == null || !file.exists()) {
            _recordingState.value = RecordingState.Error("Recording failed. Please try again.")
            return
        }

        currentAudioFile = file
        _recordingState.value = RecordingState.Processing
        callGenerateApi(file)
    }

    fun cancelRecording() {
        stopTimer()
        recorder?.cancel()
        recorder = null
        currentAudioFile = null
        _recordingState.value = RecordingState.Idle
    }

    private fun startTimer() {
        timerJob = viewModelScope.launch {
            while (true) {
                delay(1000)
                _recordingSeconds.value++
            }
        }
    }

    private fun stopTimer() {
        timerJob?.cancel()
        timerJob = null
        _recordingSeconds.value = 0
    }

    // ── API: Generate Posts ───────────────────────────────────

    private fun callGenerateApi(file: File) {
        viewModelScope.launch {
            _postsState.value = PostsState.Loading
            // Clear stale image state from any previous recording — these are keyed
            // by list index, so leftover entries would otherwise appear against
            // the wrong posts once a fresh list comes back.
            _imageGenStates.value = emptyMap()
            _selectedImages.value = emptyMap()

            try {
                val requestFile  = file.asRequestBody("audio/mp4".toMediaTypeOrNull())
                val audioPart    = MultipartBody.Part.createFormData("audio_file", file.name, requestFile)
                val tonePart     = _tone.value.toRequestBody("text/plain".toMediaTypeOrNull())
                val platformPart = _platform.value.toRequestBody("text/plain".toMediaTypeOrNull())
                val userIdPart   = _userId.value.toRequestBody("text/plain".toMediaTypeOrNull())

                val response = RetrofitClient.api.generatePost(
                    audioPart, tonePart, platformPart, userIdPart
                )

                if (response.isSuccessful) {
                    val body = response.body()
                    when {
                        body == null ->
                            _postsState.value = PostsState.Error("Server returned empty response")

                        body.variations.isEmpty() ->
                            _postsState.value = PostsState.Error(
                                body.message ?: "No posts generated. Speak clearly and try again."
                            )

                        // Detect fallback posts — never show them as real content
                        body.variations.all {
                            it.text.contains("fallback", ignoreCase = true) ||
                                    it.text.contains("Please try again", ignoreCase = true) ||
                                    it.text == "__FALLBACK__"
                        } ->
                            _postsState.value = PostsState.Error(
                                "⚠️ Generation failed on server.\n" +
                                        "Likely cause: Gemini API quota exhausted.\n" +
                                        "Please try again later."
                            )

                        else -> {
                            _postsState.value = PostsState.Success(
                                posts = body.variations,
                                isPartial = body.status == "partial_success"
                            )
                            // Scenario 1: user asked for an image up front via the toggle —
                            // auto-generate for the top-scored post as soon as it lands.
                            if (_wantsImage.value && body.variations.isNotEmpty()) {
                                generateImageForPost(0, body.variations.first().text, autoSelectFirst = true)
                            }
                        }
                    }
                } else {
                    val raw = response.errorBody()?.string() ?: ""
                    val detail = try {
                        JSONObject(raw).getString("detail")
                    } catch (_: Exception) {
                        "Server error ${response.code()}"
                    }
                    _postsState.value = PostsState.Error(detail)
                }

            } catch (e: SocketTimeoutException) {
                _postsState.value = PostsState.Error(
                    "Request timed out. The server may be starting up — please try again."
                )
            } catch (e: Exception) {
                _postsState.value = PostsState.Error("Network error: ${e.message}")
            } finally {
                _recordingState.value = RecordingState.Idle
                file.delete()
                currentAudioFile = null
            }
        }
    }

    // ── API: Publish Post ─────────────────────────────────────

    fun publishPost(postText: String) {
        viewModelScope.launch {
            _publishState.value = PublishState.Publishing
            try {
                val response = RetrofitClient.api.publishPost(
                    platform = _platform.value,
                    postText = postText
                )
                if (response.isSuccessful) {
                    val msg = response.body()?.message ?: "Published successfully!"
                    _publishState.value = PublishState.Success(msg)
                } else {
                    val raw = response.errorBody()?.string() ?: ""
                    val detail = try { JSONObject(raw).getString("detail") }
                    catch (_: Exception) { "Publish failed (${response.code()})" }
                    _publishState.value = PublishState.Error(detail)
                }
            } catch (e: Exception) {
                _publishState.value = PublishState.Error("Network error: ${e.message}")
            }
        }
    }

    // ── API: Schedule Post ────────────────────────────────────

    fun schedulePost(postText: String, scheduledTime: String) {
        viewModelScope.launch {
            _publishState.value = PublishState.Publishing
            try {
                val request = ConfirmPostRequest(
                    platform       = _platform.value,
                    text           = postText,
                    scheduled_time = scheduledTime,
                    user_id        = _userId.value
                )
                val response = RetrofitClient.api.confirmPost(request)

                if (response.isSuccessful) {
                    // Make the time human readable: "2026-04-11T18:30:00" → "2026-04-11 at 18:30"
                    val displayTime = scheduledTime
                        .replace("T", " at ")
                        .take(19)
                    _publishState.value = PublishState.Success(
                        "Scheduled for $displayTime ✅"
                    )
                } else {
                    val raw = response.errorBody()?.string() ?: ""
                    val detail = try { JSONObject(raw).getString("detail") }
                    catch (_: Exception) { "Scheduling failed (${response.code()})" }
                    _publishState.value = PublishState.Error(detail)
                }
            } catch (e: Exception) {
                _publishState.value = PublishState.Error("Network error: ${e.message}")
            }
        }
    }

    // ── API: Generate Image ───────────────────────────────────

    /**
     * Triggers image generation for the post at [postIndex].
     * Used both by the manual per-card "Image" button (autoSelectFirst = false)
     * and by the toggle-driven auto-generation on the top post (autoSelectFirst = true).
     */
    fun generateImageForPost(postIndex: Int, text: String, autoSelectFirst: Boolean = false) {
        viewModelScope.launch {
            _imageGenStates.update { it + (postIndex to ImageGenState.Loading) }
            try {
                val response = RetrofitClient.api.generateImageForPost(
                    postText = text,
                    platform = _platform.value,
                    method = "stock",
                    numOptions = 3,
                    returnBase64 = true
                )
                if (response.isSuccessful) {
                    val options = response.body()?.images ?: emptyList()
                    if (options.isNotEmpty()) {
                        _imageGenStates.update { it + (postIndex to ImageGenState.Success(options)) }
                        if (autoSelectFirst) {
                            _selectedImages.update { it + (postIndex to options.first()) }
                        }
                    } else if (autoSelectFirst) {
                        // Silently skip for the automatic path — don't surface an error
                        // for something the user didn't explicitly ask for this time.
                        _imageGenStates.update { it - postIndex }
                    } else {
                        _imageGenStates.update {
                            it + (postIndex to ImageGenState.Error("No images found"))
                        }
                    }
                } else {
                    if (autoSelectFirst) {
                        _imageGenStates.update { it - postIndex }
                    } else {
                        val raw = response.errorBody()?.string() ?: ""
                        val detail = try { JSONObject(raw).getString("detail") }
                        catch (_: Exception) { "Image generation failed (${response.code()})" }
                        _imageGenStates.update { it + (postIndex to ImageGenState.Error(detail)) }
                    }
                }
            } catch (e: Exception) {
                if (autoSelectFirst) {
                    _imageGenStates.update { it - postIndex }
                } else {
                    _imageGenStates.update {
                        it + (postIndex to ImageGenState.Error("Network error: ${e.message}"))
                    }
                }
            }
        }
    }

    fun selectImage(postIndex: Int, option: GenerateImageOption) {
        _selectedImages.update { it + (postIndex to option) }
    }

    fun clearImage(postIndex: Int) {
        _selectedImages.update { it - postIndex }
    }

    // ── Save to gallery / Share ────────────────────────────────

    private suspend fun decodeBitmap(option: GenerateImageOption): Bitmap = withContext(Dispatchers.IO) {
        option.image_base64?.let { b64 ->
            val bytes = Base64.decode(b64, Base64.DEFAULT)
            return@withContext BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                ?: throw IllegalStateException("Could not decode image data")
        }
        option.image_url?.let { url ->
            URL(url).openStream().use { input ->
                return@withContext BitmapFactory.decodeStream(input)
                    ?: throw IllegalStateException("Could not decode image from URL")
            }
        }
        throw IllegalStateException("Image has neither a URL nor image data")
    }

    fun saveImageToGallery(context: Context, option: GenerateImageOption) {
        viewModelScope.launch {
            _mediaActionState.value = MediaActionState.Working
            try {
                val bitmap = decodeBitmap(option)
                withContext(Dispatchers.IO) {
                    val fileName = "voicetopost_${System.currentTimeMillis()}.png"
                    val values = ContentValues().apply {
                        put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
                        put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                            put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
                        }
                    }
                    val uri: Uri = context.contentResolver.insert(
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values
                    ) ?: throw IllegalStateException("Could not create gallery entry")

                    context.contentResolver.openOutputStream(uri)?.use { out ->
                        bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                    } ?: throw IllegalStateException("Could not open gallery entry for writing")
                }
                _mediaActionState.value = MediaActionState.Success("Saved to gallery")
            } catch (e: Exception) {
                _mediaActionState.value = MediaActionState.Error("Save failed: ${e.message}")
            }
        }
    }

    fun shareImage(context: Context, option: GenerateImageOption) {
        viewModelScope.launch {
            _mediaActionState.value = MediaActionState.Working
            try {
                val bitmap = decodeBitmap(option)
                val uri = withContext(Dispatchers.IO) {
                    val cacheDir = File(context.cacheDir, "shared_images").apply { mkdirs() }
                    val file = File(cacheDir, "share_${System.currentTimeMillis()}.png")
                    FileOutputStream(file).use { out ->
                        bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                    }
                    FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
                }
                val intent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                    type = "image/png"
                    putExtra(android.content.Intent.EXTRA_STREAM, uri)
                    addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                context.startActivity(
                    android.content.Intent.createChooser(intent, "Share post image").apply {
                        addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                )
                _mediaActionState.value = MediaActionState.Idle
            } catch (e: Exception) {
                _mediaActionState.value = MediaActionState.Error("Share failed: ${e.message}")
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        stopTimer()
        recorder?.cancel()
        currentAudioFile?.delete()
    }
}