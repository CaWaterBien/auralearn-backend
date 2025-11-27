<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class ActivitySubmission extends Model
{
    use HasFactory;

    protected $fillable = [
        'user_id',
        'activity_id',
        'submitted_code',
        'generated_output',
        'validation_results',
        'instruction_compliance',
        'completion_status',
        'attempt_number',
        'score',
        'time_spent_minutes',
        'feedback',
        'error_details',
        'code_explanation',
        'explanation_word_count',
        'explanation_required',
        'explanation_analysis',
        'is_completed',
        'submitted_at',
        'completed_at'
    ];

    protected $casts = [
        'validation_results' => 'array',
        'instruction_compliance' => 'array',
        'error_details' => 'array',
        'explanation_analysis' => 'array',
        'is_completed' => 'boolean',
        'explanation_required' => 'boolean',
        'submitted_at' => 'datetime',
        'completed_at' => 'datetime'
    ];

    /**
     * Get the user that made this submission
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Get the activity this submission is for
     */
    public function activity(): BelongsTo
    {
        return $this->belongsTo(Activity::class);
    }

    /**
     * Scope to get only completed submissions
     */
    public function scopeCompleted($query)
    {
        return $query->whereRaw('is_completed IS TRUE');
    }

    /**
     * Scope to get only passed submissions
     */
    public function scopePassed($query)
    {
        return $query->where('completion_status', 'passed');
    }

    /**
     * Scope to get latest attempt for user/activity combination
     */
    public function scopeLatestAttempt($query, $userId, $activityId)
    {
        return $query->where('user_id', $userId)
                    ->where('activity_id', $activityId)
                    ->orderBy('attempt_number', 'desc')
                    ->first();
    }

    /**
     * Get all attempts for a user/activity combination
     */
    public function scopeAllAttempts($query, $userId, $activityId)
    {
        return $query->where('user_id', $userId)
                    ->where('activity_id', $activityId)
                    ->orderBy('attempt_number', 'asc');
    }

    /**
     * Check if this submission meets minimum requirements for completion
     */
    public function meetsMinimumRequirements(): bool
    {
        $validationResults = $this->validation_results;
        
        if (!$validationResults || !is_array($validationResults)) {
            return false;
        }

        // Check if all required elements are present
        if (isset($validationResults['required_elements_check']) && !$validationResults['required_elements_check']) {
            return false;
        }

        // Check if structure requirements are met
        if (isset($validationResults['structure_check']) && !$validationResults['structure_check']) {
            return false;
        }

        // Check if instruction compliance is adequate
        $compliance = $this->instruction_compliance;
        if (!$compliance || !is_array($compliance)) {
            return false;
        }

        $completedInstructions = array_sum(array_values($compliance));
        $totalInstructions = count($compliance);
        
        // Require at least 80% instruction compliance
        return $totalInstructions > 0 && ($completedInstructions / $totalInstructions) >= 0.8;
    }

    /**
     * Calculate completion percentage based on validation results
     */
    public function getCompletionPercentage(): int
    {
        $validationResults = $this->validation_results;
        if (!$validationResults || !is_array($validationResults)) {
            return 0;
        }

        $checks = [
            'doctype_check' => 10,
            'required_elements_check' => 25,
            'structure_check' => 25,
            'content_check' => 20,
            'instruction_compliance_check' => 20
        ];

        $totalScore = 0;
        foreach ($checks as $check => $weight) {
            if (isset($validationResults[$check]) && $validationResults[$check]) {
                $totalScore += $weight;
            }
        }

        return $totalScore;
    }
}
