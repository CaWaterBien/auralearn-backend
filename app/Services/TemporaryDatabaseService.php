<?php

namespace App\Services;

use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Facades\Log;

/**
 * Temporary file-based database service to handle missing PostgreSQL driver
 * This is a temporary workaround until PostgreSQL PHP extension is installed
 */
class TemporaryDatabaseService
{
    private $storageDir = 'temp_db';

    public function __construct()
    {
        if (!Storage::disk('local')->exists($this->storageDir)) {
            Storage::disk('local')->makeDirectory($this->storageDir);
        }
    }

    public function storeSubmission($data)
    {
        $filename = $this->storageDir . '/submissions.json';
        $submissions = $this->getStoredData($filename);
        
        $data['id'] = count($submissions) + 1;
        $data['created_at'] = now()->toISOString();
        $data['updated_at'] = now()->toISOString();
        
        $submissions[] = $data;
        
        Storage::disk('local')->put($filename, json_encode($submissions, JSON_PRETTY_PRINT));
        
        Log::info('Stored submission temporarily', ['id' => $data['id']]);
        
        return $data;
    }

    public function getSubmissionStatus($userId, $activityId)
    {
        $filename = $this->storageDir . '/submissions.json';
        $submissions = $this->getStoredData($filename);
        
        $userSubmissions = array_filter($submissions, function($sub) use ($userId, $activityId) {
            return $sub['user_id'] == $userId && $sub['activity_id'] == $activityId;
        });
        
        if (empty($userSubmissions)) {
            return [
                'total_attempts' => 0,
                'remaining_attempts' => null, // Unlimited attempts
                'best_score' => 0,
                'is_completed' => false,
                'latest_submission' => null
            ];
        }
        
        $latestSubmission = end($userSubmissions);
        $bestScore = max(array_column($userSubmissions, 'score'));
        
        return [
            'total_attempts' => count($userSubmissions),
            'remaining_attempts' => null, // Unlimited attempts - tracking only for AuraBot display
            'best_score' => $bestScore,
            'is_completed' => $latestSubmission['is_completed'] ?? false,
            'latest_submission' => $latestSubmission
        ];
    }

    public function logActivity($data)
    {
        $filename = $this->storageDir . '/activity_logs.json';
        $logs = $this->getStoredData($filename);
        
        $data['id'] = count($logs) + 1;
        $data['created_at'] = now()->toISOString();
        
        $logs[] = $data;
        
        Storage::disk('local')->put($filename, json_encode($logs, JSON_PRETTY_PRINT));
    }

    private function getStoredData($filename)
    {
        if (!Storage::disk('local')->exists($filename)) {
            return [];
        }
        
        $content = Storage::disk('local')->get($filename);
        return json_decode($content, true) ?: [];
    }

    /**
     * Clear all temporary data for a specific user and activity
     */
    public function clearUserActivityData($userId, $activityId)
    {
        $filename = $this->storageDir . '/submissions.json';
        $submissions = $this->getStoredData($filename);
        
        // Remove all submissions for this user/activity combination
        $filteredSubmissions = array_filter($submissions, function($sub) use ($userId, $activityId) {
            return !($sub['user_id'] == $userId && $sub['activity_id'] == $activityId);
        });
        
        Storage::disk('local')->put($filename, json_encode(array_values($filteredSubmissions), JSON_PRETTY_PRINT));
        
        Log::info('Cleared temporary data for user/activity', [
            'user_id' => $userId, 
            'activity_id' => $activityId,
            'removed_count' => count($submissions) - count($filteredSubmissions)
        ]);
    }

    /**
     * Clear all temporary data (for testing purposes)
     */
    public function clearAllData()
    {
        $files = ['submissions.json', 'activity_logs.json'];
        foreach ($files as $file) {
            $fullPath = $this->storageDir . '/' . $file;
            if (Storage::disk('local')->exists($fullPath)) {
                Storage::disk('local')->delete($fullPath);
                Log::info('Cleared temporary file: ' . $file);
            }
        }
    }
}