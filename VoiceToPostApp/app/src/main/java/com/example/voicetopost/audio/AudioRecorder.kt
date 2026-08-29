package com.example.voicetopost.audio



import android.content.Context
import android.media.MediaRecorder
import android.os.Build
import java.io.File

class AudioRecorder(private val context: Context) {

    private var recorder: MediaRecorder? = null
    private var outputFile: File? = null
    private var startTimeMs = 0L

    val recordingDurationMs: Long
        get() = if (startTimeMs > 0) System.currentTimeMillis() - startTimeMs else 0L

    fun start(): File {
        val file = File(
            context.cacheDir,
            "vtp_audio_${System.currentTimeMillis()}.mp4"   // .mp4 extension — correct for MPEG-4 container
        )
        outputFile = file

        recorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            MediaRecorder(context)
        } else {
            @Suppress("DEPRECATION")
            MediaRecorder()
        }

        recorder!!.apply {
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            setAudioSamplingRate(16000)     // 16kHz — standard for speech-to-text (NOT 44100)
            setAudioEncodingBitRate(32000)  // 32kbps — enough for clear speech, smaller upload
            setOutputFile(file.absolutePath)
            prepare()
            start()
        }

        startTimeMs = System.currentTimeMillis()
        return file
    }

    fun stop(): File? {
        return try {
            recorder?.apply {
                stop()
                release()
            }
            recorder = null
            startTimeMs = 0L
            outputFile
        } catch (e: Exception) {
            release()
            null
        }
    }

    fun cancel() {
        release()
        outputFile?.delete()
        outputFile = null
    }

    private fun release() {
        try { recorder?.stop() } catch (_: Exception) {}
        try { recorder?.release() } catch (_: Exception) {}
        recorder = null
        startTimeMs = 0L
    }
}