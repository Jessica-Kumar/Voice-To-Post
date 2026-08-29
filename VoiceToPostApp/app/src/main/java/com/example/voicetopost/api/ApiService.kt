package com.example.voicetopost.api



import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Response
import retrofit2.http.*

// ── Response Models ───────────────────────────────────────────

data class AuthResponse(
    val status: String,
    val user_id: String,
    val access_token: String,
    val token_type: String
)

data class PostVariation(
    val text: String,
    val score: Double
)

data class GeneratePostResponse(
    val status: String,
    val variations: List<PostVariation>,
    val total_generated: Int,
    val attempts_used: Int,
    val message: String?
)

data class PublishResponse(
    val status: String,
    val message: String? = null,
    val post_id: String? = null
)

data class ParseScheduleResponse(
    val parsed_time: String,
    val human_text: String
)

data class ConfirmPostRequest(
    val platform: String,
    val text: String,
    val scheduled_time: String? = null,
    val user_id: String
)

data class ConfirmPostResponse(
    val status: String,
    val message: String? = null
)

data class GenerateImageOption(
    val image_url: String? = null,
    val thumbnail_url: String? = null,
    val source: String? = null,
    val keywords: List<String>? = null,
    val photographer: String? = null,
    val photographer_url: String? = null,
    val model: String? = null,
    val prompt: String? = null,
    val image_base64: String? = null
)

data class GenerateImageResponse(
    val status: String,
    val images: List<GenerateImageOption> = emptyList(),
    val count: Int = 0
)


// ── Retrofit Interface ────────────────────────────────────────

interface ApiService {

    // --- Authentication ---
    @POST("auth/device")
    suspend fun authenticateDevice(): Response<AuthResponse>

    // Generate posts from voice audio
    @Multipart
    @POST("generate-post")
    suspend fun generatePost(
        @Part audio_file: MultipartBody.Part,
        @Part("tone") tone: RequestBody,
        @Part("platform") platform: RequestBody,
        @Part("user_id") userId: RequestBody
    ): Response<GeneratePostResponse>

    // Publish a post immediately
    @FormUrlEncoded
    @POST("publish-post")
    suspend fun publishPost(
        @Field("platform") platform: String,
        @Field("post_text") postText: String
    ): Response<PublishResponse>

    // Generate a stock/AI image to go with a post
    @FormUrlEncoded
    @POST("generate-image-for-post")
    suspend fun generateImageForPost(
        @Field("post_text") postText: String,
        @Field("platform") platform: String? = null,
        @Field("method") method: String? = null,
        @Field("num_options") numOptions: Int? = null,
        @Field("return_base64") returnBase64: Boolean? = null
    ): Response<GenerateImageResponse>

    // Confirm or schedule a post
    @POST("confirm-post")
    suspend fun confirmPost(
        @Body request: ConfirmPostRequest
    ): Response<ConfirmPostResponse>

    // Parse a scheduling command from voice
    @Multipart
    @POST("parse-schedule")
    suspend fun parseSchedule(
        @Part audio_file: MultipartBody.Part
    ): Response<ParseScheduleResponse>

    // Upload brand policy PDF/TXT
    @Multipart
    @POST("upload-policy")
    suspend fun uploadPolicy(
        @Part("user_id") userId: RequestBody,
        @Part policy_file: MultipartBody.Part
    ): Response<Map<String, String>>

    // Health check
    @GET("/")
    suspend fun healthCheck(): Response<Map<String, String>>
}