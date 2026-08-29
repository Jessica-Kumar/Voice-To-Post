package com.example.voicetopost.ui.navigation

sealed class Screen(val route: String) {
    object Login   : Screen("login")
    object Home    : Screen("home")
    object Results : Screen("results")
}
