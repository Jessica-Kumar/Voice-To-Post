package com.example.voicetopost.ui.screens


import android.Manifest
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddComment
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.Forum
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Tag
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
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
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.voicetopost.ui.theme.CardBg
import com.example.voicetopost.ui.theme.DeepBg
import com.example.voicetopost.ui.theme.ErrorRed
import com.example.voicetopost.ui.theme.Mint
import com.example.voicetopost.ui.theme.MutedPurple
import com.example.voicetopost.ui.theme.Purple600
import com.example.voicetopost.ui.theme.Purple800
import com.example.voicetopost.ui.theme.WarmWhite
import com.example.voicetopost.viewmodel.MainViewModel
import com.example.voicetopost.viewmodel.PostsState
import com.example.voicetopost.viewmodel.RecordingState


data class PlatformInfo(val key: String, val label: String, val icon: ImageVector, val color: Color)

val PLATFORMS = listOf(
    PlatformInfo("linkedin",  "LinkedIn",   Icons.Filled.AddComment, Color(0xFF0A66C2)),
    PlatformInfo("twitter",   "Twitter/X",  Icons.Filled.Tag,        Color(0xFF1DA1F2)),
    PlatformInfo("discord",   "Discord",    Icons.Filled.Forum,      Color(0xFF5865F2)),
    PlatformInfo("instagram", "Instagram",  Icons.Filled.CameraAlt,  Color(0xFFE1306C)),
)

@Composable
fun HomeScreen(
    viewModel: MainViewModel,
    onPostsReady: () -> Unit,
    onGoToLogin: () -> Unit      // ← new parameter
) {
    val context = LocalContext.current

    val recordingState   by viewModel.recordingState.collectAsState()
    val postsState       by viewModel.postsState.collectAsState()
    val selectedPlatform by viewModel.platform.collectAsState()
    val selectedTone     by viewModel.tone.collectAsState()
    val recordingSeconds by viewModel.recordingSeconds.collectAsState()
    val userId           by viewModel.userId.collectAsState()
    val wantsImage       by viewModel.wantsImage.collectAsState()

    // Navigate to results when posts arrive
    LaunchedEffect(postsState) {
        if (postsState is PostsState.Success) onPostsReady()
    }

    // Mic permission
    var hasPermission by remember { mutableStateOf(false) }
    val permLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasPermission = granted }

    LaunchedEffect(Unit) {
        permLauncher.launch(Manifest.permission.RECORD_AUDIO)
        viewModel.loadUserId(context)
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Brush.verticalGradient(listOf(Color(0xFF1A0A3C), DeepBg, Color(0xFF080415))))
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(Modifier.height(56.dp))

            // ── Top bar: title + settings icon ───────────────
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "Voice To Post",
                        fontSize = 34.sp,
                        fontWeight = FontWeight.Black,
                        color = WarmWhite,
                        letterSpacing = (-0.5).sp
                    )
                    Text(
                        text = "Speak. Generate. Publish.",
                        fontSize = 13.sp,
                        color = MutedPurple,
                        letterSpacing = 1.5.sp
                    )
                }
                // Settings button → back to Login
                IconButton(onClick = onGoToLogin) {
                    Icon(
                        imageVector = Icons.Filled.Settings,
                        contentDescription = "Settings / Accounts",
                        tint = MutedPurple,
                        modifier = Modifier.size(26.dp)
                    )
                }
            }

            // ── Current user ID chip ──────────────────────────
            if (userId.isNotBlank() && userId != "user_default") {
                Spacer(Modifier.height(8.dp))
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(CardBg)
                        .clickable { onGoToLogin() }   // tap to manage accounts
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("🔗 ", fontSize = 12.sp)
                    Text(
                        text = userId,
                        fontSize = 11.sp,
                        color = MutedPurple,
                        modifier = Modifier.weight(1f)
                    )
                    Text(
                        text = "manage →",
                        fontSize = 10.sp,
                        color = Purple600
                    )
                }
            }

            Spacer(Modifier.height(36.dp))

            // ── Platform selector
            SectionLabel("Choose Platform")
            Spacer(Modifier.height(8.dp))
            PlatformRow(selected = selectedPlatform, onSelect = viewModel::selectPlatform)

            Spacer(Modifier.height(24.dp))

            // ── Tone selector
            SectionLabel("Choose Tone")
            Spacer(Modifier.height(8.dp))
            ToneRow(
                tones = viewModel.tones,
                selected = selectedTone,
                onSelect = viewModel::selectTone
            )

            Spacer(Modifier.height(20.dp))

            // ── Image generation toggle ───────────────────────
            ImageToggleRow(
                checked = wantsImage,
                onToggle = { viewModel.toggleWantsImage() }
            )

            Spacer(Modifier.height(48.dp))

            // ── Record button
            RecordButton(
                state = recordingState,
                hasPermission = hasPermission,
                seconds = recordingSeconds,
                onStart  = { viewModel.startRecording(context) },
                onStop   = { viewModel.stopAndGenerate() },
                onCancel = { viewModel.cancelRecording() }
            )

            Spacer(Modifier.height(24.dp))

            // ── Error from postsState
            if (postsState is PostsState.Error) {
                ErrorBanner((postsState as PostsState.Error).message)
            }

            // ── Loading indicator
            if (postsState is PostsState.Loading) {
                LoadingBanner()
            }

            Spacer(Modifier.height(40.dp))
        }
    }
}

// ── Section label ─────────────────────────────────────────────

@Composable
fun SectionLabel(text: String) {
    Text(
        text = text.uppercase(),
        fontSize = 11.sp,
        fontWeight = FontWeight.Bold,
        color = Purple600,
        letterSpacing = 2.sp,
        modifier = Modifier.fillMaxWidth()
    )
}

// ── Platform row ──────────────────────────────────────────────

@Composable
fun PlatformRow(selected: String, onSelect: (String) -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        PLATFORMS.forEach { p ->
            val isSelected = selected == p.key
            Box(
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(14.dp))
                    .background(
                        if (isSelected)
                            Brush.linearGradient(listOf(p.color.copy(alpha = 0.25f), CardBg))
                        else
                            Brush.linearGradient(listOf(CardBg, CardBg))
                    )
                    .border(
                        width = if (isSelected) 1.5.dp else 1.dp,
                        color = if (isSelected) p.color else WarmWhite.copy(alpha = 0.07f),
                        shape = RoundedCornerShape(14.dp)
                    )
                    .clickable { onSelect(p.key) }
                    .padding(vertical = 16.dp),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(
                        imageVector = p.icon,
                        contentDescription = p.label,
                        tint = if (isSelected) p.color else MutedPurple,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(Modifier.height(6.dp))
                    Text(
                        text = p.label,
                        fontSize = 11.sp,
                        color = if (isSelected) WarmWhite else MutedPurple,
                        fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal,
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
    }
}

// ── Tone row ──────────────────────────────────────────────────

@Composable
fun ToneRow(tones: List<String>, selected: String, onSelect: (String) -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        tones.forEach { tone ->
            val isSelected = selected == tone
            FilterChip(
                selected = isSelected,
                onClick  = { onSelect(tone) },
                label = {
                    Text(
                        text = tone.replaceFirstChar { it.uppercase() },
                        fontSize = 13.sp,
                        fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal
                    )
                },
                colors = FilterChipDefaults.filterChipColors(
                    selectedContainerColor = Purple800,
                    selectedLabelColor     = Color.White,
                    containerColor         = CardBg,
                    labelColor             = MutedPurple
                ),
                border = FilterChipDefaults.filterChipBorder(
                    enabled = true,
                    selected = isSelected,
                    selectedBorderColor = Purple600,
                    borderColor = WarmWhite.copy(alpha = 0.07f)
                )
            )
        }
    }
}

// ── Image generation toggle ─────────────────────────────────────

@Composable
fun ImageToggleRow(checked: Boolean, onToggle: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(CardBg)
            .border(1.dp, WarmWhite.copy(alpha = 0.07f), RoundedCornerShape(14.dp))
            .clickable { onToggle() }
            .padding(horizontal = 16.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = Icons.Filled.CameraAlt,
            contentDescription = null,
            tint = if (checked) Purple600 else MutedPurple,
            modifier = Modifier.size(20.dp)
        )
        Spacer(Modifier.width(10.dp))
        Column(Modifier.weight(1f)) {
            Text(
                text = "Generate an image",
                fontSize = 13.sp,
                color = WarmWhite,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = "For your top-scoring post",
                fontSize = 11.sp,
                color = MutedPurple
            )
        }
        Switch(
            checked = checked,
            onCheckedChange = { onToggle() },
            colors = SwitchDefaults.colors(
                checkedThumbColor = Color.White,
                checkedTrackColor = Purple600,
                uncheckedThumbColor = MutedPurple,
                uncheckedTrackColor = CardBg
            )
        )
    }
}

// ── Record button ─────────────────────────────────────────────

@Composable
fun RecordButton(
    state: RecordingState,
    hasPermission: Boolean,
    seconds: Int,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onCancel: () -> Unit
) {
    val isRecording  = state is RecordingState.Recording
    val isProcessing = state is RecordingState.Processing
    val isError      = state is RecordingState.Error

    val pulse = rememberInfiniteTransition(label = "pulse")
    val pulseScale by pulse.animateFloat(
        initialValue = 1f, targetValue = 1.18f,
        animationSpec = infiniteRepeatable(
            tween(700, easing = FastOutSlowInEasing),
            RepeatMode.Reverse
        ), label = "scale"
    )

    Column(horizontalAlignment = Alignment.CenterHorizontally) {

        Box(contentAlignment = Alignment.Center) {
            if (isRecording) {
                Box(
                    modifier = Modifier
                        .size(116.dp)
                        .scale(pulseScale)
                        .clip(CircleShape)
                        .background(Purple800.copy(alpha = 0.18f))
                )
            }

            Box(
                modifier = Modifier
                    .size(92.dp)
                    .clip(CircleShape)
                    .background(
                        when {
                            isRecording -> Brush.radialGradient(listOf(Color(0xFFEF4444), Color(0xFFDC2626)))
                            isError     -> Brush.radialGradient(listOf(CardBg, CardBg))
                            else        -> Brush.radialGradient(listOf(Purple600, Purple800))
                        }
                    )
                    .border(
                        width = if (isError) 1.5.dp else 0.dp,
                        color = ErrorRed,
                        shape = CircleShape
                    )
                    .clickable(enabled = hasPermission && !isProcessing) {
                        if (isRecording) onStop() else onStart()
                    },
                contentAlignment = Alignment.Center
            ) {
                when {
                    isProcessing -> CircularProgressIndicator(
                        color = WarmWhite,
                        strokeWidth = 3.dp,
                        modifier = Modifier.size(34.dp)
                    )
                    isRecording  -> Icon(
                        Icons.Filled.Stop,
                        contentDescription = "Stop recording",
                        tint = Color.White,
                        modifier = Modifier.size(38.dp)
                    )
                    else -> Icon(
                        Icons.Filled.Mic,
                        contentDescription = "Start recording",
                        tint = if (isError) MutedPurple else Color.White,
                        modifier = Modifier.size(38.dp)
                    )
                }
            }
        }

        Spacer(Modifier.height(16.dp))

        Text(
            text = when (state) {
                is RecordingState.Idle       -> if (hasPermission) "Tap to speak" else "Microphone permission needed"
                is RecordingState.Recording  -> "Recording  ${"%02d:%02d".format(seconds / 60, seconds % 60)}  •  Tap to stop"
                is RecordingState.Processing -> "Generating your posts…"
                is RecordingState.Error      -> state.message
            },
            fontSize = 14.sp,
            color = when (state) {
                is RecordingState.Recording  -> Mint
                is RecordingState.Error      -> ErrorRed
                else                         -> MutedPurple
            },
            textAlign = TextAlign.Center
        )

        AnimatedVisibility(visible = isRecording) {
            TextButton(onClick = onCancel) {
                Text("Cancel", color = MutedPurple, fontSize = 13.sp)
            }
        }
    }
}

// ── Loading banner ────────────────────────────────────────────

@Composable
fun LoadingBanner() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(CardBg)
            .padding(20.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        CircularProgressIndicator(color = Purple600, strokeWidth = 3.dp)
        Spacer(Modifier.height(12.dp))
        Text("AI is crafting your posts…", color = WarmWhite, fontSize = 14.sp)
        Text("This can take 20–30 seconds", color = MutedPurple, fontSize = 12.sp)
    }
}

// ── Error banner ──────────────────────────────────────────────

@Composable
fun ErrorBanner(message: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(Color(0xFF3B0A0A))
            .border(1.dp, ErrorRed.copy(alpha = 0.4f), RoundedCornerShape(14.dp))
            .padding(14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(Icons.Filled.ErrorOutline, contentDescription = null, tint = ErrorRed)
        Spacer(Modifier.width(10.dp))
        Text(message, color = Color(0xFFFCA5A5), fontSize = 13.sp, modifier = Modifier.weight(1f))
    }
}