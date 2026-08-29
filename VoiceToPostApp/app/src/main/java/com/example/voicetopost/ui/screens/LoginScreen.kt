package com.example.voicetopost.ui.screens

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.voicetopost.viewmodel.MainViewModel

private const val BASE_URL = "https://jessicakumar-voice-to-post.hf.space"

@Composable
fun LoginScreen(
    viewModel: MainViewModel,
    onContinue: () -> Unit
) {
    val context = LocalContext.current
    val focusManager = LocalFocusManager.current
    val focusRequester = remember { FocusRequester() }

    val connectedPlatforms by viewModel.connectedPlatforms.collectAsState()
    val userId by viewModel.userId.collectAsState()

    // Local editable state for user ID field
    var userIdInput by remember(userId) { mutableStateOf(userId) }
    var isEditingUserId by remember { mutableStateOf(false) }

    // Load saved user ID and connected platforms on first composition
    LaunchedEffect(Unit) {
        viewModel.loadUserId(context)
        viewModel.loadConnectedPlatforms(context)
    }

    // When editing starts, focus the text field
    LaunchedEffect(isEditingUserId) {
        if (isEditingUserId) focusRequester.requestFocus()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0D0D1A))
            .padding(horizontal = 28.dp)
            .padding(top = 60.dp, bottom = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // ── Logo ──────────────────────────────────────────────
        Box(
            modifier = Modifier
                .size(68.dp)
                .clip(RoundedCornerShape(20.dp))
                .background(
                    Brush.linearGradient(listOf(Color(0xFF6C5DD3), Color(0xFFA78BFA)))
                ),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                Icons.Default.GraphicEq,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(34.dp)
            )
        }

        Spacer(Modifier.height(14.dp))
        Text("Voice to Post", color = Color(0xFFE8E4FF), fontSize = 22.sp, fontWeight = FontWeight.SemiBold)
        Text("Speak once. Post everywhere.", color = Color(0xFF6B6880), fontSize = 13.sp)

        Spacer(Modifier.height(32.dp))

        // ── User ID row ───────────────────────────────────────
        Text(
            "YOUR USER ID",
            color = Color(0xFF4D4A66),
            fontSize = 9.sp,
            letterSpacing = 1.5.sp,
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(Modifier.height(6.dp))

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .background(Color(0xFF0F0F1C))
                .border(
                    width = 1.dp,
                    color = if (isEditingUserId) Color(0xFF6C5DD3) else Color(0xFF1E1E36),
                    shape = RoundedCornerShape(12.dp)
                )
                .padding(horizontal = 14.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (isEditingUserId) {
                // Editable text field
                TextField(
                    value = userIdInput,
                    onValueChange = { userIdInput = it.trim() },
                    modifier = Modifier
                        .weight(1f)
                        .focusRequester(focusRequester),
                    singleLine = true,
                    textStyle = LocalTextStyle.current.copy(
                        color = Color(0xFFA78BFA),
                        fontSize = 13.sp,
                        fontFamily = FontFamily.Monospace
                    ),
                    colors = TextFieldDefaults.colors(
                        focusedContainerColor = Color.Transparent,
                        unfocusedContainerColor = Color.Transparent,
                        focusedIndicatorColor = Color.Transparent,
                        unfocusedIndicatorColor = Color.Transparent,
                        cursorColor = Color(0xFFA78BFA)
                    ),
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                    keyboardActions = KeyboardActions(onDone = {
                        if (userIdInput.isNotBlank()) {
                            viewModel.onOAuthSuccess(context, userIdInput)
                        }
                        isEditingUserId = false
                        focusManager.clearFocus()
                    }),
                    placeholder = {
                        Text("Enter user ID", color = Color(0xFF4D4A66), fontSize = 13.sp)
                    }
                )
                // Save button
                TextButton(
                    onClick = {
                        if (userIdInput.isNotBlank()) {
                            viewModel.onOAuthSuccess(context, userIdInput)
                        }
                        isEditingUserId = false
                        focusManager.clearFocus()
                    },
                    contentPadding = PaddingValues(horizontal = 8.dp)
                ) {
                    Text("Save", color = Color(0xFF6C5DD3), fontSize = 12.sp, fontWeight = FontWeight.Medium)
                }
            } else {
                // Display mode
                Box(
                    modifier = Modifier
                        .size(34.dp)
                        .clip(CircleShape)
                        .background(
                            Brush.linearGradient(listOf(Color(0xFF6C5DD3), Color(0xFFA78BFA)))
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        userId.take(2).uppercase(),
                        color = Color.White,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }
                Spacer(Modifier.width(10.dp))
                Text(
                    userId,
                    color = Color(0xFFA78BFA),
                    fontSize = 13.sp,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.weight(1f)
                )
                // Edit icon
                Icon(
                    Icons.Default.Edit,
                    contentDescription = "Edit user ID",
                    tint = Color(0xFF4D4A66),
                    modifier = Modifier
                        .size(18.dp)
                        .clickable { isEditingUserId = true }
                )
            }
        }

        Spacer(Modifier.height(24.dp))

        // ── Platform connect section ──────────────────────────
        Text(
            "CONNECT YOUR ACCOUNTS",
            color = Color(0xFF4D4A66),
            fontSize = 9.sp,
            letterSpacing = 1.5.sp,
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(Modifier.height(8.dp))
        Text(
            "Connect to publish directly. You can also skip and copy-paste posts manually.",
            color = Color(0xFF3A3852),
            fontSize = 11.sp,
            lineHeight = 16.sp,
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(Modifier.height(12.dp))

        // LinkedIn
        PlatformConnectButton(
            label = "LinkedIn",
            sublabel = "Posts, articles & updates",
            isConnected = connectedPlatforms.contains("linkedin"),
            backgroundColor = Color(0xFF0A66C2),
            onClick = {
                val url = "$BASE_URL/auth/linkedin/login?user_id=${userId}"
                context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
            }
        )

        Spacer(Modifier.height(10.dp))

        // Twitter / X
        PlatformConnectButton(
            label = "X (Twitter)",
            sublabel = "Threads & tweets",
            isConnected = connectedPlatforms.contains("twitter"),
            backgroundColor = Color(0xFF0F141A),
            borderColor = Color(0xFF1D9BF0),
            onClick = {
                val url = "$BASE_URL/auth/twitter/login?user_id=${userId}"
                context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
            }
        )


        Spacer(Modifier.height(10.dp))

        PlatformConnectButton(
            label = "Discord",
            sublabel = "Post to a channel via OAuth",
            isConnected = connectedPlatforms.contains("discord"),
            backgroundColor = Color(0xFF5865F2),
            onClick = {
                val url = "$BASE_URL/auth/discord/login?user_id=${userId}"
                context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
            }
        )
        Spacer(Modifier.weight(1f))
        // ── Start button — ALWAYS enabled ────────────────────
        Button(
            onClick = onContinue,
            enabled = true,                        // ← always tappable
            modifier = Modifier
                .fillMaxWidth()
                .height(54.dp),
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFF6C5DD3)
            )
        ) {
            Text(
                if (connectedPlatforms.isEmpty())
                    "Continue without connecting →"
                else
                    "Start posting with voice →",
                fontWeight = FontWeight.SemiBold,
                fontSize = 15.sp,
                color = Color.White
            )
        }

        if (connectedPlatforms.isEmpty()) {
            Spacer(Modifier.height(6.dp))
            Text(
                "Connect a platform above to publish directly from the app.",
                color = Color(0xFF4D4A66),
                fontSize = 10.sp,
                lineHeight = 14.sp,
                textAlign = TextAlign.Center
            )
        }

        Spacer(Modifier.height(8.dp))
        Text(
            "By continuing, you allow Voice to Post to publish\non your behalf via each platform's official OAuth.",
            color = Color(0xFF3A3852),
            fontSize = 10.sp,
            lineHeight = 15.sp,
            textAlign = TextAlign.Center
        )
    }
}

@Composable
private fun PlatformConnectButton(
    label: String,
    sublabel: String,
    isConnected: Boolean,
    backgroundColor: Color,
    borderColor: Color = Color.Transparent,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(backgroundColor)
            .border(1.dp, borderColor, RoundedCornerShape(14.dp))
            .clickable(onClick = onClick)
            .padding(14.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column {
            Text(label, color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Medium)
            Text(sublabel, color = Color.White.copy(alpha = 0.6f), fontSize = 11.sp)
        }
        if (isConnected) {
            Text(
                "✓ connected",
                color = Color(0xFF1DB954),
                fontSize = 11.sp,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .background(Color(0x261DB954))
                    .padding(horizontal = 10.dp, vertical = 4.dp)
            )
        } else {
            Text("›", color = Color.White.copy(alpha = 0.5f), fontSize = 18.sp)
        }
    }
}