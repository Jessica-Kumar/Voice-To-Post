package com.voicetopost.data

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.core.stringSetPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import java.util.UUID

// Extension property — one DataStore per app
val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "vtp_prefs")

object UserPrefs {

    private val USER_ID_KEY              = stringPreferencesKey("user_id")
    private val ACCESS_TOKEN_KEY         = stringPreferencesKey("access_token")
    private val CONNECTED_PLATFORMS_KEY  = stringSetPreferencesKey("connected_platforms")

    // --- User ID & Token ----------------------------------------------------

    fun getUserId(context: Context): Flow<String?> {
        return context.dataStore.data.map { prefs -> prefs[USER_ID_KEY] }
    }
    
    fun getAccessToken(context: Context): Flow<String?> {
        return context.dataStore.data.map { prefs -> prefs[ACCESS_TOKEN_KEY] }
    }

    suspend fun setAuthData(context: Context, userId: String, token: String) {
        context.dataStore.edit { prefs ->
            prefs[USER_ID_KEY] = userId
            prefs[ACCESS_TOKEN_KEY] = token
        }
    }

    suspend fun setUserId(context: Context, userId: String) {
        context.dataStore.edit { prefs ->
            prefs[USER_ID_KEY] = userId
        }
    }

    // Called after LinkedIn/Twitter OAuth callback returns a real user ID
    suspend fun saveOAuthUserId(context: Context, userId: String) {
        setUserId(context, userId)
    }

    // ── Connected Platforms ───────────────────────────────────

    // Returns the set of platforms the user has connected e.g. {"linkedin", "twitter"}
    fun getConnectedPlatforms(context: Context): Flow<Set<String>> {
        return context.dataStore.data.map { prefs ->
            prefs[CONNECTED_PLATFORMS_KEY] ?: emptySet()
        }
    }

    // Adds a single platform to the persisted set
    suspend fun addConnectedPlatform(context: Context, platform: String) {
        context.dataStore.edit { prefs ->
            val current = prefs[CONNECTED_PLATFORMS_KEY] ?: emptySet()
            prefs[CONNECTED_PLATFORMS_KEY] = current + platform
        }
    }
}