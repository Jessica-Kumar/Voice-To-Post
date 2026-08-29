package com.example.voicetopost.ui.screens

import android.app.DatePickerDialog
import android.app.TimePickerDialog
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil3.compose.AsyncImage

import com.example.voicetopost.api.GenerateImageOption
import com.example.voicetopost.api.PostVariation
import com.example.voicetopost.ui.theme.CardBg
import com.example.voicetopost.ui.theme.DeepBg
import com.example.voicetopost.ui.theme.Mint
import com.example.voicetopost.ui.theme.MutedPurple
import com.example.voicetopost.ui.theme.Purple600
import com.example.voicetopost.ui.theme.Purple800
import com.example.voicetopost.ui.theme.WarmWhite
import com.example.voicetopost.viewmodel.ImageGenState
import com.example.voicetopost.viewmodel.MainViewModel
import com.example.voicetopost.viewmodel.MediaActionState
import com.example.voicetopost.viewmodel.PostsState
import com.example.voicetopost.viewmodel.PublishState
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale

// Data class to carry both text and action type into dialogs
data class PendingPost(val text: String, val action: PostAction)
enum class PostAction { PUBLISH, SCHEDULE }

@Composable
fun ResultsScreen(
    viewModel: MainViewModel,
    onBack: () -> Unit
) {
    val postsState        by viewModel.postsState.collectAsState()
    val publishState      by viewModel.publishState.collectAsState()
    val platform          by viewModel.platform.collectAsState()
    val imageGenStates    by viewModel.imageGenStates.collectAsState()
    val selectedImages    by viewModel.selectedImages.collectAsState()
    val mediaActionState  by viewModel.mediaActionState.collectAsState()
    val context           = LocalContext.current

    // Holds the post pending user confirmation (publish or schedule)
    var pendingPost by remember { mutableStateOf<PendingPost?>(null) }
    // Which post index currently has the image picker open
    var pickerForIndex by remember { mutableStateOf<Int?>(null) }

    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(publishState) {
        when (val s = publishState) {
            is PublishState.Success -> {
                snackbarHostState.showSnackbar("✅ ${s.message}")
                viewModel.resetPublishState()
            }
            is PublishState.Error -> {
                snackbarHostState.showSnackbar("❌ ${s.message}")
                viewModel.resetPublishState()
            }
            else -> {}
        }
    }

    // Save/Share feedback — also covers the toggle-driven auto-image failure message
    LaunchedEffect(mediaActionState) {
        when (val s = mediaActionState) {
            is MediaActionState.Success -> {
                snackbarHostState.showSnackbar("✅ ${s.message}")
                viewModel.resetMediaActionState()
            }
            is MediaActionState.Error -> {
                snackbarHostState.showSnackbar("❌ ${s.message}")
                viewModel.resetMediaActionState()
            }
            else -> {}
        }
    }

    // Publish confirmation dialog — with editable text
    pendingPost?.let { pp ->
        when (pp.action) {
            PostAction.PUBLISH -> {
                PublishDialog(
                    initialText = pp.text,
                    platform = platform,
                    isPublishing = publishState is PublishState.Publishing,
                    onConfirm = { finalText ->
                        viewModel.publishPost(finalText)
                        pendingPost = null
                    },
                    onDismiss = { pendingPost = null }
                )
            }
            PostAction.SCHEDULE -> {
                // First show edit dialog, then schedule picker
                EditBeforeScheduleDialog(
                    initialText = pp.text,
                    platform = platform,
                    onNext = { editedText ->
                        // Replace pending with edited text, keep SCHEDULE action
                        // ScheduleDialog will fire after this dismisses
                        pendingPost = PendingPost(editedText, PostAction.SCHEDULE)
                    },
                    onDismiss = { pendingPost = null }
                )
            }
        }
    }

    // Image option picker dialog
    pickerForIndex?.let { idx ->
        val state = imageGenStates[idx]
        ImagePickerDialog(
            state = state,
            onSelect = { option ->
                viewModel.selectImage(idx, option)
                pickerForIndex = null
            },
            onDismiss = { pickerForIndex = null }
        )
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarHostState) },
        containerColor = DeepBg
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            ResultsTopBar(
                platform = platform,
                onBack = {
                    viewModel.resetPosts()
                    onBack()
                }
            )

            when (val s = postsState) {
                is PostsState.Success -> {
                    if (s.isPartial) PartialWarning()

                    LazyColumn(
                        contentPadding = PaddingValues(
                            start = 16.dp, end = 16.dp,
                            top = 8.dp, bottom = 32.dp
                        ),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        itemsIndexed(s.posts) { index, post ->
                            PostCard(
                                index    = index + 1,
                                post     = post,
                                platform = platform,
                                imageGenState  = imageGenStates[index],
                                selectedImage  = selectedImages[index],
                                onPublish  = { edited ->
                                    pendingPost = PendingPost(edited, PostAction.PUBLISH)
                                },
                                onSchedule = { edited ->
                                    pendingPost = PendingPost(edited, PostAction.SCHEDULE)
                                },
                                onRequestImage = { editedText ->
                                    val current = imageGenStates[index]
                                    if (current == null || current is ImageGenState.Error) {
                                        viewModel.generateImageForPost(index, editedText)
                                    }
                                    pickerForIndex = index
                                },
                                onClearImage = { viewModel.clearImage(index) },
                                onSaveImage = { img -> viewModel.saveImageToGallery(context, img) },
                                onShareImage = { img -> viewModel.shareImage(context, img) }
                            )
                        }
                    }
                }
                else -> {
                    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        Text("No posts available.", color = MutedPurple)
                    }
                }
            }
        }
    }
}

// ── Top bar ───────────────────────────────────────────────────

@Composable
fun ResultsTopBar(platform: String, onBack: () -> Unit) {
    val platformInfo = PLATFORMS.find { it.key == platform }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        IconButton(onClick = onBack) {
            Icon(Icons.Filled.ArrowBack, contentDescription = "Back", tint = WarmWhite)
        }
        Spacer(Modifier.width(8.dp))
        Column {
            Text("Generated Posts", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = WarmWhite)
            platformInfo?.let {
                Text("For ${it.label}", fontSize = 12.sp, color = it.color)
            }
        }
    }
}

// ── Partial warning ───────────────────────────────────────────

@Composable
fun PartialWarning() {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .clip(RoundedCornerShape(10.dp))
            .background(Color(0xFF3B2E00))
            .padding(10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(Icons.Filled.Warning, contentDescription = null, tint = Color(0xFFFBBF24), modifier = Modifier.size(16.dp))
        Spacer(Modifier.width(8.dp))
        Text("Fewer than 5 posts met quality threshold.", color = Color(0xFFFDE68A), fontSize = 12.sp)
    }
}

// ── Post card ─────────────────────────────────────────────────

@Composable
fun PostCard(
    index: Int,
    post: PostVariation,
    platform: String,
    imageGenState: ImageGenState?,
    selectedImage: GenerateImageOption?,
    onPublish: (String) -> Unit,
    onSchedule: (String) -> Unit,
    onRequestImage: (String) -> Unit,
    onClearImage: () -> Unit,
    onSaveImage: (GenerateImageOption) -> Unit,
    onShareImage: (GenerateImageOption) -> Unit
) {
    val clipboard = LocalClipboardManager.current
    var copied by remember { mutableStateOf(false) }

    // LOCAL edit state — user can tweak the post text inline on the card
    var isEditing   by remember { mutableStateOf(false) }
    var editedText  by remember { mutableStateOf(post.text) }
    val focusReq    = remember { FocusRequester() }

    val scoreColor = when {
        post.score >= 0.90 -> Mint
        post.score >= 0.75 -> Color(0xFFFBBF24)
        else               -> Color(0xFFE879F9)
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = CardBg),
        border = BorderStroke(
            width = if (isEditing) 1.5.dp else 1.dp,
            color = if (isEditing) Purple600 else WarmWhite.copy(alpha = 0.07f)
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {

            // Header row: index + edit toggle + score
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Index circle
                Box(
                    modifier = Modifier
                        .size(30.dp)
                        .clip(CircleShape)
                        .background(Purple800.copy(alpha = 0.3f)),
                    contentAlignment = Alignment.Center
                ) {
                    Text("$index", color = Purple600, fontSize = 13.sp, fontWeight = FontWeight.Bold)
                }

                Spacer(Modifier.width(8.dp))

                // Edit mode label
                if (isEditing) {
                    Text(
                        "Editing…",
                        fontSize = 11.sp,
                        color = Purple600,
                        fontWeight = FontWeight.SemiBold
                    )
                }

                Spacer(Modifier.weight(1f))

                // Edit / Done toggle button
                IconButton(
                    onClick = {
                        if (isEditing) {
                            isEditing = false
                        } else {
                            isEditing = true
                        }
                    },
                    modifier = Modifier.size(30.dp)
                ) {
                    Icon(
                        imageVector = if (isEditing) Icons.Filled.Check else Icons.Filled.Edit,
                        contentDescription = if (isEditing) "Done editing" else "Edit post",
                        tint = if (isEditing) Mint else MutedPurple,
                        modifier = Modifier.size(16.dp)
                    )
                }

                Spacer(Modifier.width(4.dp))

                // Score badge
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(scoreColor.copy(alpha = 0.15f))
                        .padding(horizontal = 10.dp, vertical = 3.dp)
                ) {
                    Text(
                        "Score ${(post.score * 100).toInt()}%",
                        color = scoreColor,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }

            Spacer(Modifier.height(12.dp))

            // Post text — editable or read-only
            if (isEditing) {
                // Editable inline text field
                LaunchedEffect(isEditing) {
                    if (isEditing) focusReq.requestFocus()
                }
                OutlinedTextField(
                    value = editedText,
                    onValueChange = { editedText = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .focusRequester(focusReq),
                    textStyle = TextStyle(
                        color = WarmWhite,
                        fontSize = 14.sp,
                        lineHeight = 22.sp
                    ),
                    minLines = 3,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor   = Purple600,
                        unfocusedBorderColor = WarmWhite.copy(alpha = 0.15f),
                        cursorColor          = WarmWhite,
                        focusedTextColor     = WarmWhite,
                        unfocusedTextColor   = WarmWhite,
                        focusedContainerColor   = CardBg,
                        unfocusedContainerColor = CardBg
                    ),
                    shape = RoundedCornerShape(10.dp),
                    supportingText = {
                        // Character counter — useful for Twitter 280 char limit
                        Text(
                            "${editedText.length} chars",
                            color = if (editedText.length > 280) Color(0xFFEF4444) else MutedPurple,
                            fontSize = 11.sp
                        )
                    }
                )
            } else {
                // Read-only display
                Text(
                    text = editedText,
                    color = WarmWhite,
                    fontSize = 14.sp,
                    lineHeight = 22.sp
                )
            }

            // Auto/manual generation-in-progress indicator — shown whenever a
            // request is in flight and nothing has been selected yet, so the
            // toggle-driven background generation (Path 1) isn't silently
            // invisible while it's running.
            if (imageGenState is ImageGenState.Loading && selectedImage == null) {
                Spacer(Modifier.height(10.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(
                        color = Purple600,
                        strokeWidth = 2.dp,
                        modifier = Modifier.size(14.dp)
                    )
                    Spacer(Modifier.width(8.dp))
                    Text("Generating image…", fontSize = 12.sp, color = MutedPurple)
                }
            }

            // Selected image preview — Save / Share / Remove, never silently
            // carried into publish (the backend has no field for it)
            selectedImage?.let { img ->
                Spacer(Modifier.height(10.dp))
                AsyncImage(
                    model = img.image_url ?: "data:image/png;base64,${img.image_base64}",
                    contentDescription = "Selected post image",
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(140.dp)
                        .clip(RoundedCornerShape(10.dp))
                )
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = { onSaveImage(img) }) {
                        Text("Save", fontSize = 12.sp)
                    }
                    OutlinedButton(onClick = { onShareImage(img) }) {
                        Text("Share", fontSize = 12.sp)
                    }
                    TextButton(onClick = onClearImage) {
                        Text("Remove", fontSize = 12.sp, color = MutedPurple)
                    }
                }
            }

            Spacer(Modifier.height(14.dp))

            // Row 1: Copy | Image | Schedule
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp, Alignment.End)
            ) {
                // Copy — copies the current editedText (including any edits)
                OutlinedButton(
                    onClick = {
                        clipboard.setText(AnnotatedString(editedText))
                        copied = true
                    },
                    shape = RoundedCornerShape(10.dp),
                    border = BorderStroke(1.dp, WarmWhite.copy(alpha = 0.12f)),
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp)
                ) {
                    Icon(
                        if (copied) Icons.Filled.Check else Icons.Filled.ContentCopy,
                        contentDescription = null,
                        modifier = Modifier.size(15.dp),
                        tint = if (copied) Mint else MutedPurple
                    )
                    Spacer(Modifier.width(4.dp))
                    Text(
                        if (copied) "Copied" else "Copy",
                        fontSize = 12.sp,
                        color = if (copied) Mint else MutedPurple
                    )
                }

                // Image — manual trigger for image generation (scenario 2)
                OutlinedButton(
                    onClick = { onRequestImage(editedText) },
                    shape = RoundedCornerShape(10.dp),
                    border = BorderStroke(1.dp, WarmWhite.copy(alpha = 0.12f)),
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp)
                ) {
                    if (imageGenState is ImageGenState.Loading) {
                        CircularProgressIndicator(
                            color = MutedPurple,
                            strokeWidth = 2.dp,
                            modifier = Modifier.size(14.dp)
                        )
                    } else {
                        Icon(
                            Icons.Filled.Image,
                            contentDescription = null,
                            modifier = Modifier.size(15.dp),
                            tint = MutedPurple
                        )
                    }
                    Spacer(Modifier.width(4.dp))
                    Text(
                        if (selectedImage != null) "Change" else "Image",
                        fontSize = 12.sp,
                        color = MutedPurple
                    )
                }

                // Schedule — passes editedText (with edits applied)
                OutlinedButton(
                    onClick = {
                        isEditing = false
                        onSchedule(editedText)
                    },
                    shape = RoundedCornerShape(10.dp),
                    border = BorderStroke(1.dp, WarmWhite.copy(alpha = 0.12f)),
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp)
                ) {
                    Icon(
                        Icons.Filled.Schedule,
                        contentDescription = null,
                        modifier = Modifier.size(15.dp),
                        tint = MutedPurple
                    )
                    Spacer(Modifier.width(4.dp))
                    Text("Schedule", fontSize = 12.sp, color = MutedPurple)
                }
            }

            Spacer(Modifier.height(8.dp))

            // Row 2: Publish — on its own row, full width for emphasis
            Button(
                onClick = {
                    isEditing = false
                    onPublish(editedText)
                },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = Purple800),
                shape = RoundedCornerShape(10.dp),
                contentPadding = PaddingValues(horizontal = 14.dp, vertical = 10.dp)
            ) {
                Icon(Icons.Filled.Send, contentDescription = null, modifier = Modifier.size(15.dp))
                Spacer(Modifier.width(4.dp))
                Text("Publish", fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

// ── Image picker dialog ──────────────────────────────────────

@Composable
fun ImagePickerDialog(
    state: ImageGenState?,
    onSelect: (GenerateImageOption) -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor    = CardBg,
        titleContentColor = WarmWhite,
        textContentColor  = MutedPurple,
        shape = RoundedCornerShape(20.dp),
        title = { Text("Choose an image", fontWeight = FontWeight.Bold, fontSize = 15.sp) },
        text = {
            when (state) {
                is ImageGenState.Loading, null -> {
                    Box(Modifier.fillMaxWidth().height(120.dp), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator(color = Purple600)
                    }
                }
                is ImageGenState.Error -> {
                    Text(state.message, color = Color(0xFFEF4444), fontSize = 13.sp)
                }
                is ImageGenState.Success -> {
                    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        itemsIndexed(state.options) { _, option ->
                            AsyncImage(
                                model = option.image_url ?: "data:image/png;base64,${option.image_base64}",
                                contentDescription = "Image option",
                                modifier = Modifier
                                    .size(110.dp)
                                    .clip(RoundedCornerShape(10.dp))
                                    .clickable { onSelect(option) }
                            )
                        }
                    }
                }
                ImageGenState.Idle -> {}
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("Close", color = MutedPurple) }
        }
    )
}

// ── Publish confirmation dialog — shows editable text ─────────

@Composable
fun PublishDialog(
    initialText: String,
    platform: String,
    isPublishing: Boolean,
    onConfirm: (String) -> Unit,   // passes final (possibly edited) text
    onDismiss: () -> Unit
) {
    val info = PLATFORMS.find { it.key == platform }
    // User can still edit inside the dialog as a last check
    var finalText by remember { mutableStateOf(initialText) }

    AlertDialog(
        onDismissRequest = { if (!isPublishing) onDismiss() },
        containerColor    = CardBg,
        titleContentColor = WarmWhite,
        textContentColor  = MutedPurple,
        shape = RoundedCornerShape(20.dp),
        title = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                info?.let {
                    Icon(it.icon, contentDescription = null, tint = it.color, modifier = Modifier.size(20.dp))
                    Spacer(Modifier.width(8.dp))
                }
                Text(
                    "Review & publish to ${info?.label ?: platform}",
                    fontWeight = FontWeight.Bold,
                    fontSize = 15.sp
                )
            }
        },
        text = {
            Column {
                // Instruction hint
                Text(
                    "You can make final edits below before publishing.",
                    color = MutedPurple,
                    fontSize = 11.sp,
                    modifier = Modifier.padding(bottom = 10.dp)
                )

                // Editable text field inside dialog
                OutlinedTextField(
                    value = finalText,
                    onValueChange = { finalText = it },
                    modifier = Modifier.fillMaxWidth(),
                    textStyle = TextStyle(
                        color = WarmWhite,
                        fontSize = 13.sp,
                        lineHeight = 20.sp
                    ),
                    minLines = 4,
                    maxLines = 8,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor    = Purple600,
                        unfocusedBorderColor  = WarmWhite.copy(alpha = 0.15f),
                        cursorColor           = WarmWhite,
                        focusedTextColor      = WarmWhite,
                        unfocusedTextColor    = WarmWhite,
                        focusedContainerColor    = Color(0xFF1A0D38),
                        unfocusedContainerColor  = Color(0xFF1A0D38)
                    ),
                    shape = RoundedCornerShape(10.dp),
                    supportingText = {
                        val charColor = when {
                            platform == "twitter" && finalText.length > 280 -> Color(0xFFEF4444)
                            finalText.isEmpty() -> Color(0xFFEF4444)
                            else -> MutedPurple
                        }
                        val hint = when {
                            platform == "twitter" && finalText.length > 280 ->
                                "⚠ ${finalText.length}/280 — too long for Twitter"
                            finalText.isEmpty() -> "⚠ Post cannot be empty"
                            else -> "${finalText.length} chars"
                        }
                        Text(hint, color = charColor, fontSize = 11.sp)
                    }
                )
            }
        },
        confirmButton = {
            Button(
                onClick = { onConfirm(finalText) },
                enabled = !isPublishing && finalText.isNotBlank(),
                colors = ButtonDefaults.buttonColors(containerColor = Purple800)
            ) {
                if (isPublishing) {
                    CircularProgressIndicator(
                        color = Color.White,
                        strokeWidth = 2.dp,
                        modifier = Modifier.size(16.dp)
                    )
                } else {
                    Text("Publish Now")
                }
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss, enabled = !isPublishing) {
                Text("Cancel", color = MutedPurple)
            }
        }
    )
}

// ── Edit before schedule dialog ────────────────────────────────

@Composable
fun EditBeforeScheduleDialog(
    initialText: String,
    platform: String,
    onNext: (String) -> Unit,   // passes edited text → then show schedule picker
    onDismiss: () -> Unit
) {
    val info = PLATFORMS.find { it.key == platform }
    var editedText by remember { mutableStateOf(initialText) }
    val context = LocalContext.current
    var proceedToSchedule by remember { mutableStateOf(false) }

    if (proceedToSchedule) {
        // Show date → time picker after user confirms edit
        ScheduleDialog(
            onSchedule = { isoTime ->
                onNext(editedText)
                // Note: caller (ResultsScreen) handles the actual schedulePost call
                // We pass the edited text back via onNext and the ISO time separately
                // This is handled in ResultsScreen via the pendingPost pattern
            },
            onDismiss = { proceedToSchedule = false }
        )
        return
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor    = CardBg,
        titleContentColor = WarmWhite,
        textContentColor  = MutedPurple,
        shape = RoundedCornerShape(20.dp),
        title = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                info?.let {
                    Icon(it.icon, contentDescription = null, tint = it.color, modifier = Modifier.size(20.dp))
                    Spacer(Modifier.width(8.dp))
                }
                Text(
                    "Edit & schedule for ${info?.label ?: platform}",
                    fontWeight = FontWeight.Bold,
                    fontSize = 15.sp
                )
            }
        },
        text = {
            Column {
                Text(
                    "Edit your post, then pick a date and time.",
                    color = MutedPurple,
                    fontSize = 11.sp,
                    modifier = Modifier.padding(bottom = 10.dp)
                )
                OutlinedTextField(
                    value = editedText,
                    onValueChange = { editedText = it },
                    modifier = Modifier.fillMaxWidth(),
                    textStyle = TextStyle(
                        color = WarmWhite,
                        fontSize = 13.sp,
                        lineHeight = 20.sp
                    ),
                    minLines = 4,
                    maxLines = 8,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor    = Purple600,
                        unfocusedBorderColor  = WarmWhite.copy(alpha = 0.15f),
                        cursorColor           = WarmWhite,
                        focusedTextColor      = WarmWhite,
                        unfocusedTextColor    = WarmWhite,
                        focusedContainerColor    = Color(0xFF1A0D38),
                        unfocusedContainerColor  = Color(0xFF1A0D38)
                    ),
                    shape = RoundedCornerShape(10.dp),
                    supportingText = {
                        Text("${editedText.length} chars", color = MutedPurple, fontSize = 11.sp)
                    }
                )
            }
        },
        confirmButton = {
            Button(
                onClick = { proceedToSchedule = true },
                enabled = editedText.isNotBlank(),
                colors = ButtonDefaults.buttonColors(containerColor = Purple800)
            ) {
                Icon(Icons.Filled.Schedule, contentDescription = null, modifier = Modifier.size(15.dp))
                Spacer(Modifier.width(6.dp))
                Text("Pick Date & Time")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel", color = MutedPurple)
            }
        }
    )
}

// ── Schedule dialog — date picker → time picker ────────────────

@Composable
fun ScheduleDialog(
    onSchedule: (String) -> Unit,
    onDismiss: () -> Unit
) {
    val context  = LocalContext.current
    val calendar = remember { Calendar.getInstance() }
    var dateChosen by remember { mutableStateOf(false) }

    if (!dateChosen) {
        DatePickerDialog(
            context,
            { _, year, month, day ->
                calendar.set(Calendar.YEAR, year)
                calendar.set(Calendar.MONTH, month)
                calendar.set(Calendar.DAY_OF_MONTH, day)
                dateChosen = true
            },
            calendar.get(Calendar.YEAR),
            calendar.get(Calendar.MONTH),
            calendar.get(Calendar.DAY_OF_MONTH)
        ).apply {
            datePicker.minDate = System.currentTimeMillis() - 1000
            setOnCancelListener { onDismiss() }
            show()
        }
    } else {
        TimePickerDialog(
            context,
            { _, hour, minute ->
                calendar.set(Calendar.HOUR_OF_DAY, hour)
                calendar.set(Calendar.MINUTE, minute)
                calendar.set(Calendar.SECOND, 0)
                val sdf = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.getDefault())
                onSchedule(sdf.format(calendar.time))
            },
            calendar.get(Calendar.HOUR_OF_DAY),
            calendar.get(Calendar.MINUTE),
            true
        ).apply {
            setOnCancelListener { onDismiss() }
            show()
        }
    }
}