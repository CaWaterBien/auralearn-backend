<?php

namespace App\Http\Controllers;

use App\Models\User;
use App\Models\ActivitySubmission;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Validator;
use Illuminate\Support\Facades\DB;

class UserSettingsController extends Controller
{
    /**
     * Update user settings (all preferences)
     */
    public function updateSettings(Request $request)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        $validator = Validator::make($request->all(), [
            // General
            'theme' => ['sometimes', 'in:light,dark'],
            'language' => ['sometimes', 'string', 'max:10'],
            
            // Notifications
            'emailNotifications' => ['sometimes', 'boolean'],
            'pushNotifications' => ['sometimes', 'boolean'],
            'weeklyDigest' => ['sometimes', 'boolean'],
            'achievementAlerts' => ['sometimes', 'boolean'],
            'reminderAlerts' => ['sometimes', 'boolean'],
            
            // Privacy
            'profileVisibility' => ['sometimes', 'in:public,friends,private'],
            'showProgress' => ['sometimes', 'boolean'],
            'showBadges' => ['sometimes', 'boolean'],
            'allowMessages' => ['sometimes', 'boolean'],
            
            // Learning
            'dailyGoal' => ['sometimes', 'integer', 'min:5', 'max:480'],
            'difficultyLevel' => ['sometimes', 'in:beginner,intermediate,advanced,expert'],
            'autoSave' => ['sometimes', 'boolean'],
            'skipIntros' => ['sometimes', 'boolean'],
            'showHints' => ['sometimes', 'boolean'],
            
            // Accessibility
            'fontSize' => ['sometimes', 'in:small,medium,large,extra-large'],
            'reducedMotion' => ['sometimes', 'boolean'],
            'highContrast' => ['sometimes', 'boolean'],
            'soundEffects' => ['sometimes', 'boolean'],
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            // Get current preferences or default ones
            $currentPreferences = $user->preferences ?? $user->getDefaultPreferences();
            
            // Merge with new settings
            $updatedPreferences = array_merge($currentPreferences, $request->all());
            
            // Update user preferences
            $user->update(['preferences' => $updatedPreferences]);

            return response()->json([
                'message' => 'Settings updated successfully',
                'preferences' => $user->preferences,
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Settings update failed',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get user's storage usage
     */
    public function getStorageUsage(Request $request)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        try {
            $usedBytes = $user->calculateStorageUsage();
            $maxBytes = 100 * 1024 * 1024; // 100 MB
            
            $usedMB = round($usedBytes / (1024 * 1024), 2);
            $maxMB = 100;
            $percentage = round(($usedBytes / $maxBytes) * 100, 2);

            return response()->json([
                'usedBytes' => $usedBytes,
                'usedMB' => $usedMB,
                'maxMB' => $maxMB,
                'percentage' => $percentage,
                'percentageFormatted' => $percentage . '%',
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to get storage usage',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Export all user data
     */
    public function exportData(Request $request)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        try {
            // Gather all user data
            $userData = [
                'user_info' => [
                    'id' => $user->id,
                    'full_name' => $user->full_name,
                    'email' => $user->email,
                    'join_date' => $user->join_date,
                    'avatar' => $user->avatar,
                ],
                'progress' => $user->progress,
                'preferences' => $user->preferences,
                'activity_submissions' => [],
                'export_date' => now()->toDateTimeString(),
            ];

            // Get all activity submissions
            $submissions = $user->activitySubmissions()
                ->with('activity')
                ->orderBy('created_at', 'desc')
                ->get();

            foreach ($submissions as $submission) {
                $userData['activity_submissions'][] = [
                    'activity_id' => $submission->activity_id,
                    'activity_title' => $submission->activity->title ?? 'Unknown',
                    'attempt_number' => $submission->attempt_number,
                    'is_completed' => $submission->is_completed,
                    'score' => $submission->score,
                    'submitted_code' => $submission->submitted_code,
                    'feedback' => $submission->feedback,
                    'created_at' => $submission->created_at,
                    'completed_at' => $submission->completed_at,
                ];
            }

            return response()->json([
                'message' => 'Data exported successfully',
                'data' => $userData,
                'filename' => 'auralearn-data-' . $user->id . '-' . now()->format('Y-m-d') . '.json'
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Data export failed',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Clear all user data (progress, submissions, etc.)
     */
    public function clearData(Request $request)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        // Require confirmation
        $validator = Validator::make($request->all(), [
            'confirm' => ['required', 'boolean', 'accepted'],
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Confirmation required to clear data',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            DB::beginTransaction();

            // Delete all activity submissions
            $user->activitySubmissions()->delete();
            
            // Reset progress to default
            $user->update([
                'progress' => $user->getDefaultProgress(),
            ]);

            DB::commit();

            return response()->json([
                'message' => 'All data cleared successfully',
                'progress' => $user->progress,
            ]);

        } catch (\Exception $e) {
            DB::rollBack();
            
            return response()->json([
                'message' => 'Data clearing failed',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Reset settings to defaults
     */
    public function resetSettings(Request $request)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        try {
            $defaultPreferences = $user->getDefaultPreferences();
            
            $user->update(['preferences' => $defaultPreferences]);

            return response()->json([
                'message' => 'Settings reset to defaults',
                'preferences' => $user->preferences,
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Settings reset failed',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get current user settings
     */
    public function getSettings(Request $request)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        // Ensure preferences exist, otherwise use defaults
        $preferences = $user->preferences ?? $user->getDefaultPreferences();

        return response()->json([
            'preferences' => $preferences,
        ]);
    }
}

