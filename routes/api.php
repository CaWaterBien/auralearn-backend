<?php

use Illuminate\Support\Facades\Route;
use Illuminate\Support\Facades\Storage;
use App\Http\Controllers\AdminAuthController;
use App\Http\Controllers\AdminDashboardController;
use App\Http\Controllers\AdminUserController;
use App\Http\Controllers\UserAuthController;
use App\Http\Controllers\UserSettingsController;
use App\Http\Controllers\FileUploadController;

use App\Http\Controllers\CourseController;
use App\Http\Controllers\LessonController;
use App\Http\Controllers\TopicController;
use App\Http\Controllers\CodeExampleController;
use App\Http\Controllers\ActivityController;
use App\Http\Controllers\AuraBotController;
use App\Http\Controllers\AchievementController;
use App\Http\Controllers\DashboardController;
//hello world
// AuraBot RAG API Routes (public access for frontend)
Route::prefix('aurabot')->group(function () {
    Route::post('ask', [AuraBotController::class, 'askQuestion']);
    Route::get('session-status', [AuraBotController::class, 'getSessionStatus']);
    Route::get('conversation-history', [AuraBotController::class, 'getConversationHistory']);
    Route::get('health', [AuraBotController::class, 'healthCheck']);
    
    // Admin routes for session management
    Route::middleware(['auth:admin'])->group(function () {
        Route::post('reset-session', [AuraBotController::class, 'resetSession']);
    });
});

// Public test routes (no authentication)
Route::get('test-simple', function () {
    return response()->json(['status' => 'ok', 'message' => 'API is working']);
});
Route::post('test-upload/image', [FileUploadController::class, 'uploadImage']);
Route::post('test-upload/video', [FileUploadController::class, 'uploadVideo']);

Route::prefix('admin')->middleware('admin_api')->group(function () {
    Route::post('login', [AdminAuthController::class, 'login']);

    Route::middleware('auth:admin')->group(function () {
        Route::get('me', [AdminAuthController::class, 'me']);
        Route::post('logout', [AdminAuthController::class, 'logout']);

        Route::get('overview', [AdminDashboardController::class, 'overview']);
        Route::get('users', [AdminDashboardController::class, 'users']);
        Route::get('logs', [AdminDashboardController::class, 'logs']);
        Route::get('settings', [AdminDashboardController::class, 'settings']);

        // User Management Routes
        Route::prefix('user-management')->group(function () {
            Route::get('/', [AdminUserController::class, 'index']);
            Route::post('/', [AdminUserController::class, 'store']);
            Route::get('/stats', [AdminUserController::class, 'stats']);
            Route::get('/{id}', [AdminUserController::class, 'show']);
            Route::put('/{id}', [AdminUserController::class, 'update']);
            Route::delete('/{id}', [AdminUserController::class, 'destroy']);
            Route::patch('/{id}/toggle-status', [AdminUserController::class, 'toggleStatus']);
        });

        // Course Management Routes
        Route::prefix('courses')->group(function () {
            Route::get('/', [CourseController::class, 'index']);
            Route::post('/', [CourseController::class, 'store']);
            Route::get('/{id}', [CourseController::class, 'show']);
            Route::put('/{id}', [CourseController::class, 'update']);
            Route::delete('/{id}', [CourseController::class, 'destroy']);
            Route::patch('/{id}/toggle-published', [CourseController::class, 'togglePublished']);
            
            // Lessons
            Route::get('/{courseId}/lessons', [LessonController::class, 'index']);
            Route::post('/{courseId}/lessons', [LessonController::class, 'store']);
            Route::get('/{courseId}/lessons/{lessonId}', [LessonController::class, 'show']);
            Route::put('/{courseId}/lessons/{lessonId}', [LessonController::class, 'update']);
            Route::delete('/{courseId}/lessons/{lessonId}', [LessonController::class, 'destroy']);
            Route::post('/{courseId}/lessons/reorder', [LessonController::class, 'reorder']);
        });

        // Topics
        Route::prefix('lessons/{lessonId}/topics')->group(function () {
            Route::get('/', [TopicController::class, 'index']);
            Route::post('/', [TopicController::class, 'store']);
            Route::get('/{topicId}', [TopicController::class, 'show']);
            Route::put('/{topicId}', [TopicController::class, 'update']);
            Route::delete('/{topicId}', [TopicController::class, 'destroy']);
            Route::post('/reorder', [TopicController::class, 'reorder']);
        });

        // Activities
        Route::prefix('lessons/{lessonId}/activities')->group(function () {
            Route::get('/', [ActivityController::class, 'index']);
            Route::post('/', [ActivityController::class, 'store']);
            Route::get('/stats', [ActivityController::class, 'stats']);
            Route::get('/{activityId}', [ActivityController::class, 'show']);
            Route::put('/{activityId}', [ActivityController::class, 'update']);
            Route::delete('/{activityId}', [ActivityController::class, 'destroy']);
            Route::post('/reorder', [ActivityController::class, 'reorder']);
            
            // Activity submission management
            Route::get('/{activityId}/submissions', [ActivityController::class, 'getSubmissions']);
        });
        
        // Activity submission detail routes
        Route::prefix('submissions')->group(function () {
            Route::get('/{submissionId}', [ActivityController::class, 'getSubmissionDetail']);
            Route::put('/{submissionId}/status', [ActivityController::class, 'updateSubmissionStatus']);
        });
        
        // Analytics routes
        Route::prefix('analytics')->group(function () {
            Route::get('activities/{activityId}', [ActivityController::class, 'getActivityAnalytics']);
            Route::get('activities/{activityId}/patterns', [ActivityController::class, 'getActivityPatterns']);
            Route::get('activities/{activityId}/errors', [ActivityController::class, 'getCommonErrors']);
            Route::get('dashboard', [ActivityController::class, 'getAnalyticsDashboard']);
        });

        // Code Examples
        Route::prefix('code-examples')->group(function () {
            Route::get('/', [CodeExampleController::class, 'index']);
            Route::post('/', [CodeExampleController::class, 'store']);
            Route::get('/{id}', [CodeExampleController::class, 'show']);
            Route::put('/{id}', [CodeExampleController::class, 'update']);
            Route::delete('/{id}', [CodeExampleController::class, 'destroy']);
            Route::post('/{id}/duplicate', [CodeExampleController::class, 'duplicate']);
        });

        // File Upload Routes
        Route::prefix('upload')->group(function () {
            Route::post('image', [FileUploadController::class, 'uploadImage']);
            Route::post('video', [FileUploadController::class, 'uploadVideo']);
            Route::delete('file', [FileUploadController::class, 'deleteFile']);
            Route::get('file-info', [FileUploadController::class, 'getFileInfo']);
            
            // Test route for debugging
            Route::get('test', function () {
                return response()->json(['message' => 'Upload routes are working!', 'timestamp' => now()]);
            });
            
            // Test file serving
            Route::get('test-file-serving', function () {
                $testFiles = [];
                
                // Check if storage link exists
                $publicStoragePath = public_path('storage');
                $storageExists = is_link($publicStoragePath) || is_dir($publicStoragePath);
                
                // List recent uploaded files
                $uploadPath = storage_path('app/public/uploads/images/topic_image');
                $recentFiles = [];
                
                if (is_dir($uploadPath)) {
                    $files = glob($uploadPath . '/*/*/*');
                    $recentFiles = array_slice(array_reverse($files), 0, 5);
                }
                
                return response()->json([
                    'storage_link_exists' => $storageExists,
                    'public_storage_path' => $publicStoragePath,
                    'upload_directory' => $uploadPath,
                    'recent_files' => $recentFiles,
                    'storage_url_example' => Storage::url('uploads/images/topic_image/test.png'),
                    'cors_test' => 'CORS enabled for file serving'
                ]);
            });
            
            // Direct file serving route for debugging
            Route::get('serve-file/{path}', function ($path) {
                $decodedPath = urldecode($path);
                $fullPath = storage_path('app/public/' . $decodedPath);
                
                if (!file_exists($fullPath)) {
                    return response()->json(['error' => 'File not found', 'path' => $fullPath], 404);
                }
                
                $mimeType = mime_content_type($fullPath);
                return response()->file($fullPath, [
                    'Content-Type' => $mimeType,
                    'Access-Control-Allow-Origin' => '*',
                    'Access-Control-Allow-Methods' => 'GET, OPTIONS',
                    'Access-Control-Allow-Headers' => 'Content-Type'
                ]);
            })->where('path', '.*');
        });

    });
});

// User Authentication Routes
Route::prefix('user')->middleware('user_api')->group(function () {
    // Public routes
    Route::post('register', [UserAuthController::class, 'register']);
    Route::post('login', [UserAuthController::class, 'login']);
    
    // Protected routes
    Route::middleware('auth:web')->group(function () {
        Route::get('me', [UserAuthController::class, 'me']);
        Route::post('logout', [UserAuthController::class, 'logout']);
        Route::put('profile', [UserAuthController::class, 'updateProfile']);
        
        // Settings Routes
        Route::get('settings', [UserSettingsController::class, 'getSettings']);
        Route::put('settings', [UserSettingsController::class, 'updateSettings']);
        Route::post('settings/reset', [UserSettingsController::class, 'resetSettings']);
        Route::get('settings/storage', [UserSettingsController::class, 'getStorageUsage']);
        Route::get('settings/export', [UserSettingsController::class, 'exportData']);
        Route::post('settings/clear-data', [UserSettingsController::class, 'clearData']);
    });
});

// Public Course Routes (for users to access)
Route::prefix('courses')->group(function () {
    Route::get('/', [CourseController::class, 'index']);
    Route::get('/slug/{slug}', [CourseController::class, 'getBySlug']);
    Route::get('/{courseId}/lessons/{lessonId}', [LessonController::class, 'show']);
    
    // Progress tracking (requires authentication)
    Route::middleware('auth:web')->group(function () {
        Route::post('/{courseId}/lessons/{lessonId}/complete', [LessonController::class, 'markComplete']);
        Route::post('/code-examples/{id}/validate', [CodeExampleController::class, 'validate']);
    });
});

// Public Activity Routes (for direct activity access)
Route::get('activities/{activityId}', [ActivityController::class, 'showById']);

// Activity Submission Routes - USER ISOLATION HANDLED IN CONTROLLER
Route::post('activities/{activityId}/submit', [ActivityController::class, 'submitActivity']);
Route::get('activities/{activityId}/status', [ActivityController::class, 'getSubmissionStatus']);
Route::delete('activities/{activityId}/clear-data', [ActivityController::class, 'clearTemporaryData']);
Route::delete('activities/clear-all-data', [ActivityController::class, 'clearTemporaryData']);

// Achievement Routes - Public access with user_id parameter
Route::prefix('achievements')->group(function () {
    Route::get('/', [AchievementController::class, 'index']);
    Route::get('/stats', [AchievementController::class, 'stats']);
    Route::get('/by-course', [AchievementController::class, 'byCourse']);
});

// User Progress Routes
Route::get('user/progress', [AchievementController::class, 'userProgress']);

// Course-specific achievements
Route::get('courses/{courseId}/achievements', [AchievementController::class, 'courseAchievements']);

// Dashboard Stats Route
Route::get('dashboard/stats', [DashboardController::class, 'getDashboardStats']); 