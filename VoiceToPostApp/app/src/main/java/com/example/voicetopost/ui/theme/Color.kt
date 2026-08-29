package com.example.voicetopost.ui.theme

import androidx.compose.ui.graphics.Color
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

val Purple80 = Color(0xFFD0BCFF)
val PurpleGrey80 = Color(0xFFCCC2DC)
val Pink80 = Color(0xFFEFB8C8)

val Purple40 = Color(0xFF6650a4)
val PurpleGrey40 = Color(0xFF625b71)
val Pink40 = Color(0xFF7D5260)




val Purple800   = Color(0xFF7C3AED)
val Purple600   = Color(0xFF9F67FF)
val DeepBg      = Color(0xFF0D0820)
val SurfaceBg   = Color(0xFF160A30)
val CardBg      = Color(0xFF1E1040)
val WarmWhite   = Color(0xFFF5F0FF)
val MutedPurple = Color(0xFF9B8CBF)
val Mint        = Color(0xFF34D399)
val ErrorRed    = Color(0xFFEF4444)

private val ColorScheme = darkColorScheme(
    primary         = Purple800,
    onPrimary       = Color.White,
    secondary       = Purple600,
    background      = DeepBg,
    surface         = SurfaceBg,
    surfaceVariant  = CardBg,
    onBackground    = WarmWhite,
    onSurface       = WarmWhite,
    onSurfaceVariant= MutedPurple,
    error           = ErrorRed,
)

@Composable
fun VoiceToPostTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = ColorScheme,
        content = content
    )
}