package com.example.voicetopost

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.lifecycle.ViewModelProvider
import androidx.navigation.NavController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.voicetopost.ui.navigation.Screen
import com.example.voicetopost.ui.screens.HomeScreen
import com.example.voicetopost.ui.screens.LoginScreen
import com.example.voicetopost.ui.screens.ResultsScreen
import com.example.voicetopost.ui.theme.VoiceToPostTheme
import com.example.voicetopost.viewmodel.MainViewModel
import com.voicetopost.data.UserPrefs

import kotlinx.coroutines.flow.first
import com.example.voicetopost.api.RetrofitClient

class MainActivity : ComponentActivity() {

    private lateinit var viewModel: MainViewModel
    private var navController: NavController? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        viewModel = ViewModelProvider(this)[MainViewModel::class.java]

        setContent {
            VoiceToPostTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val navController = rememberNavController().also {
                        this.navController = it
                    }

                    // Determine start destination based on saved platforms
                    var startDestination by remember { mutableStateOf<String?>(null) }

                    LaunchedEffect(Unit) {
                        // 1. Silent Zero-Friction Auth
                        val token = UserPrefs.getAccessToken(this@MainActivity).first()
                        if (token.isNullOrEmpty()) {
                            try {
                                val response = RetrofitClient.api.authenticateDevice()
                                if (response.isSuccessful && response.body() != null) {
                                    val authData = response.body()!!
                                    UserPrefs.setAuthData(this@MainActivity, authData.user_id, authData.access_token)
                                    RetrofitClient.currentToken = authData.access_token
                                    Log.d("Auth", "Device authenticated successfully")
                                }
                            } catch (e: Exception) {
                                Log.e("Auth", "Device auth failed", e)
                            }
                        } else {
                            RetrofitClient.currentToken = token
                        }

                        // 2. Check login state for navigation
                        UserPrefs.getConnectedPlatforms(this@MainActivity).collect { platforms ->
                            if (startDestination == null) {
                                startDestination = if (platforms.isNotEmpty()) Screen.Home.route
                                else Screen.Login.route
                            }
                        }
                    }

                    if (startDestination == null) {
                        // Show spinner while reading prefs
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            CircularProgressIndicator(color = Color(0xFF6C5DD3))
                        }
                    } else {
                        NavHost(
                            navController = navController,
                            startDestination = startDestination!!
                        ) {
                            composable(Screen.Login.route) {
                                LoginScreen(
                                    viewModel = viewModel,
                                    onContinue = {
                                        navController.navigate(Screen.Home.route) {
                                            popUpTo(Screen.Login.route) { inclusive = true }
                                        }
                                    }
                                )
                            }
                            composable(Screen.Home.route) {
                                HomeScreen(
                                    viewModel = viewModel,
                                    onPostsReady = {
                                        navController.navigate(Screen.Results.route)
                                    },
                                    onGoToLogin = {
                                        navController.navigate(Screen.Login.route)
                                    }
                                )
                            }
                            composable(Screen.Results.route) {
                                ResultsScreen(
                                    viewModel = viewModel,
                                    onBack = {
                                        navController.popBackStack()
                                    }
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)

        val data = intent.data ?: return
        Log.d("VoiceToPost", "onNewIntent: $data")

        if (data.scheme == "yourapp" && data.host == "callback") {
            val userId   = data.getQueryParameter("user_id")
            val platform = data.getQueryParameter("platform") ?: return

            Log.d("VoiceToPost", "OAuth callback — userId=$userId, platform=$platform")

            if (!userId.isNullOrBlank() && userId != viewModel.userId.value) {
                viewModel.onOAuthSuccess(this, userId)
            }

            if (platform.isNotBlank()) {
                viewModel.markPlatformConnected(this, platform)
            }

            // Navigate to Home after OAuth completes
            navController?.navigate(Screen.Home.route) {
                popUpTo(Screen.Login.route) { inclusive = true }
            }
        }
    }
}