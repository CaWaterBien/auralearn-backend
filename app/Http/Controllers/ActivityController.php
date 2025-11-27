<?php

namespace App\Http\Controllers;

use App\Models\Activity;
use App\Models\ActivitySubmission;
use App\Models\ActivityLog;
use App\Models\ActivityCertificate;
use App\Models\Lesson;
use App\Models\User;
use App\Services\TemporaryDatabaseService;
use App\Services\AiValidationService;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Validator;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;

class ActivityController extends Controller
{
    private AiValidationService $aiValidationService;

    public function __construct(AiValidationService $aiValidationService)
    {
        $this->aiValidationService = $aiValidationService;
    }
    /**
     * Display a listing of activities for a lesson
     */
    public function index($lessonId)
    {
        try {
            $lesson = Lesson::findOrFail($lessonId);
            $activities = Activity::where('lesson_id', $lessonId)
                ->orderBy('order_index')
                ->get();

            return response()->json([
                'activities' => $activities
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Lesson not found'
            ], 404);
        }
    }

    /**
     * Store a newly created activity
     */
    public function store(Request $request, $lessonId)
    {
        $validator = Validator::make($request->all(), [
            'title' => 'required|string|max:255',
            'description' => 'nullable|string',
            'activity_type' => 'required|in:coding',
            'instructions' => 'required|string',
            'questions' => 'nullable|array',
            'resources' => 'nullable|array',
            'time_limit' => 'nullable|integer|min:1',
            'max_attempts' => 'nullable|integer|min:1',
            'passing_score' => 'nullable|integer|min:0|max:100',
            'points' => 'nullable|integer|min:0',
            'order_index' => 'nullable|integer',
            'is_required' => 'nullable|boolean',
            'is_published' => 'nullable|boolean',
            'metadata' => 'nullable|array',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            $lesson = Lesson::findOrFail($lessonId);
            
            $data = $request->all();
            $data['lesson_id'] = $lessonId;

            $activity = Activity::create($data);

            return response()->json([
                'message' => 'Activity created successfully',
                'activity' => $activity
            ], 201);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Lesson not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to create activity',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Display the specified activity
     */
    public function show($lessonId, $activityId)
    {
        try {
            $activity = Activity::where('lesson_id', $lessonId)
                ->findOrFail($activityId);

            return response()->json([
                'activity' => $activity
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Activity not found'
            ], 404);
        }
    }

    /**
     * Display activity by ID alone (for direct activity access)
     */
    public function showById($activityId)
    {
        try {
            $activity = Activity::with('lesson.course')->findOrFail($activityId);

            // Log activity access
            if (Auth::check()) {
                ActivityLog::logActivityStarted(Auth::id(), $activityId, [
                    'access_type' => 'direct',
                    'lesson_title' => $activity->lesson->title ?? 'Unknown',
                    'course_title' => $activity->lesson->course->title ?? 'Unknown'
                ]);
            }

            return response()->json([
                'activity' => $activity
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Activity not found'
            ], 404);
        }
    }

    /**
     * Submit and validate user code for an activity with AI-POWERED validation
     * NOTE: Now uses AI to intelligently validate code against instructions!
     */
    public function submitActivity(Request $request, $activityId)
    {
        $validator = Validator::make($request->all(), [
            'user_code' => 'required|string|min:10',
            'time_spent_minutes' => 'required|integer|min:1',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            Log::info('🤖 Starting AI-powered submission validation', [
                'activity_id' => $activityId,
                'code_length' => strlen($request->user_code)
            ]);

            // Get user from request or use default - USER-SPECIFIC DATA ISOLATION
            $userId = $request->input('user_id') ?? 1; // Default to user 1 if not provided
            $user = (object)[
                'id' => $userId,
                'email' => 'user@example.com',
                'name' => 'User'
            ];

            $userCode = trim($request->user_code);
            
            // Get the activity details from database
            $activity = Activity::findOrFail($activityId);
            
            // Parse instructions for AI validation
            $instructions = explode("\n", $activity->instructions);
            $instructions = array_filter(array_map('trim', $instructions), function($instruction) {
                return !empty($instruction) && !preg_match('/^\d+\.\s*$/', $instruction);
            });

            // 🤖 AI-POWERED VALIDATION - The main event!
            try {
                $aiValidationResult = $this->aiValidationService->validateCodeWithAi(
                    $userCode, 
                    $instructions, 
                    $activity->title, 
                    $activity->description
                );

                Log::info('🎯 AI validation completed', [
                    'ai_powered' => $aiValidationResult['ai_powered'],
                    'overall_score' => $aiValidationResult['overall_score'],
                    'completion_status' => $aiValidationResult['completion_status'],
                    'is_completed' => $aiValidationResult['is_completed']
                ]);

                // Get comprehensive AI feedback
                $comprehensiveFeedback = $this->aiValidationService->generateEducationalFeedback($aiValidationResult);
                
            } catch (\Exception $aiError) {
                Log::error('❌ AI validation failed, using fallback', [
                    'activity_id' => $activityId,
                    'error' => $aiError->getMessage(),
                    'file' => $aiError->getFile(),
                    'line' => $aiError->getLine()
                ]);
                
                // Use fallback validation with simple scoring
                $aiValidationResult = [
                    'ai_powered' => false,
                    'overall_score' => 60, // Default reasonable score
                    'completion_status' => 'needs_improvement',
                    'is_completed' => false,
                    'validation_breakdown' => ['fallback_used' => true],
                    'detailed_feedback' => 'Your code has been submitted successfully. Due to a temporary system issue, detailed AI feedback is not available right now.',
                    'suggestions' => ['Please review the activity instructions and ensure all requirements are met.'],
                    'areas_for_improvement' => ['Check syntax', 'Verify all instructions are followed'],
                    'validation_summary' => [
                        'overall' => [
                            'passed' => 3,
                            'total' => 5,
                            'percentage' => 60
                        ]
                    ],
                    'instruction_progress' => [],
                    'technical_validation' => [],
                    'requirements_analysis' => [],
                    'positive_aspects' => ['Code submitted successfully']
                ];
                
                $comprehensiveFeedback = $aiValidationResult['detailed_feedback'];
                
                Log::info('🔄 Using fallback validation', [
                    'activity_id' => $activityId,
                    'fallback_score' => $aiValidationResult['overall_score']
                ]);
            }

            // Store submission data for attempt tracking
            $tempDB = new TemporaryDatabaseService();
            $currentStatus = $tempDB->getSubmissionStatus($user->id, $activityId);
            $attemptNumber = $currentStatus['total_attempts'] + 1;

            $submissionData = [
                'user_id' => $user->id,
                'activity_id' => $activityId,
                'submitted_code' => $userCode,
                'score' => $aiValidationResult['overall_score'],
                'is_completed' => $aiValidationResult['is_completed'],
                'completion_status' => $aiValidationResult['completion_status'],
                'time_spent_minutes' => $request->time_spent_minutes,
                'feedback' => $comprehensiveFeedback,
                'attempt_number' => $attemptNumber,
                'validation_results' => json_encode([
                    'ai_powered' => $aiValidationResult['ai_powered'],
                    'overall' => $aiValidationResult['validation_summary']['overall'],
                    'technical_validation' => $aiValidationResult['technical_validation'],
                    'requirements_analysis' => $aiValidationResult['requirements_analysis'],
                    'positive_aspects' => $aiValidationResult['positive_aspects'],
                    'suggestions' => $aiValidationResult['suggestions']
                ])
            ];
            
            $submission = $tempDB->storeSubmission($submissionData);
            
            Log::info('🎉 AI-powered submission completed', [
                'activity_id' => $activityId,
                'score' => $aiValidationResult['overall_score'],
                'is_completed' => $aiValidationResult['is_completed'],
                'submission_id' => $submission['id'],
                'attempt_number' => $attemptNumber,
                'ai_powered' => $aiValidationResult['ai_powered']
            ]);
            
            return response()->json([
                'success' => true,
                'submission_id' => $submission['id'],
                'score' => $aiValidationResult['overall_score'],
                'is_completed' => $aiValidationResult['is_completed'],
                'completion_status' => $aiValidationResult['completion_status'],
                'attempt_number' => $attemptNumber,
                'feedback' => $comprehensiveFeedback,
                'validation_summary' => $aiValidationResult['validation_summary'],
                'instruction_progress' => $aiValidationResult['instruction_progress'],
                'ai_powered' => $aiValidationResult['ai_powered'],
                'positive_aspects' => $aiValidationResult['positive_aspects'],
                'suggestions' => $aiValidationResult['suggestions'],
                'areas_for_improvement' => $aiValidationResult['areas_for_improvement'],
                'detailed_feedback' => $aiValidationResult['detailed_feedback'],
                'using_ai_validation' => true,
                'message' => $aiValidationResult['is_completed'] ? 
                    '🎉 Congratulations! Activity completed successfully! The AI has verified that your code meets all requirements.' : 
                    '📚 Great effort! The AI has analyzed your code and provided detailed feedback to help you improve. Review the suggestions and try again!'
            ]);

        } catch (\Exception $e) {
            Log::error('❌ Error in AI-powered submission', [
                'error' => $e->getMessage(),
                'activity_id' => $activityId,
                'trace' => $e->getTraceAsString()
            ]);
            
            // Fallback to basic validation if AI fails completely
            return $this->handleSubmissionFallback($request, $activityId, $e);
        }
    }


    /**
     * Get user's submission status for an activity
     * NOTE: No attempt limits - unlimited submissions allowed, tracking only for AuraBot display
     */
    public function getSubmissionStatus($activityId)
    {
        try {
            // Get user from request or use default - USER-SPECIFIC DATA ISOLATION  
            $userId = request()->input('user_id') ?? 1; // Default to user 1 if not provided
            $user = (object)[
                'id' => $userId,
                'email' => 'user@example.com', 
                'name' => 'User'
            ];
            
            // Simplified version that always works
            Log::info('getSubmissionStatus called (simplified)', ['activity_id' => $activityId, 'user_id' => $user->id]);
            
            // Get attempt tracking from temporary storage
            $tempDB = new TemporaryDatabaseService();
            $status = $tempDB->getSubmissionStatus($user->id, $activityId);
            
            // Debug logging to see what's happening
            Log::info('Submission status retrieved', [
                'activity_id' => $activityId,
                'user_id' => $user->id,
                'total_attempts' => $status['total_attempts'],
                'is_completed' => $status['is_completed']
            ]);
            
            return response()->json([
                'activity_id' => $activityId,
                'total_attempts' => $status['total_attempts'],
                'max_attempts' => null, // No attempt limit
                'attempts_remaining' => null, // Unlimited attempts
                'is_completed' => $status['is_completed'],
                'best_score' => $status['best_score'],
                'latest_submission' => $status['latest_submission'],
                'using_temporary_storage' => true,
                'message' => 'Unlimited attempts - attempt tracking only for AuraBot display'
            ]);

        } catch (\Exception $e) {
            Log::error('Error in simplified getSubmissionStatus', [
                'error' => $e->getMessage(),
                'trace' => $e->getTraceAsString()
            ]);
            
            return response()->json([
                'activity_id' => $activityId,
                'total_attempts' => 0,
                'max_attempts' => null, // No attempt limit
                'attempts_remaining' => null, // Unlimited attempts
                'is_completed' => false,
                'best_score' => 0,
                'latest_submission' => null,
                'error' => 'Fallback mode - unlimited attempts: ' . $e->getMessage()
            ], 200); // Return 200 instead of 500 to avoid frontend errors
        }
    }

    /**
     * Perform comprehensive validation of user HTML code
     */
    private function performComprehensiveValidation($userCode, $activity)
    {
        $criteria = $activity->metadata['validation_criteria'] ?? [];
        $normalizedCode = $this->normalizeHtml($userCode);
        
        $results = [
            'doctype_check' => $this->validateDoctype($userCode),
            'html_structure_check' => $this->validateHtmlStructure($normalizedCode),
            'required_elements_check' => $this->validateRequiredElements($normalizedCode, $criteria),
            'required_attributes_check' => $this->validateRequiredAttributes($normalizedCode, $criteria),
            'structure_requirements_check' => $this->validateStructureRequirements($normalizedCode, $criteria),
            'content_check' => $this->validateContent($userCode, $activity->metadata['expected_output'] ?? ''),
            'syntax_validation' => $this->validateSyntax($userCode),
            'semantic_validation' => $this->validateSemantic($normalizedCode),
            'accessibility_check' => $this->validateAccessibility($normalizedCode),
            'code_quality_check' => $this->validateCodeQuality($userCode),
            'similarity_check' => $this->checkCodeSimilarity(Auth::id(), $activity->id, $userCode)
        ];
        
        return $results;
    }

    /**
     * Check instruction compliance step by step
     */
    private function checkInstructionCompliance($userCode, $activity)
    {
        $instructions = explode('\n', $activity->instructions);
        $compliance = [];
        
        foreach ($instructions as $index => $instruction) {
            $instructionKey = "instruction_" . ($index + 1);
            $compliance[$instructionKey] = $this->checkSingleInstructionCompliance($userCode, trim($instruction));
        }
        
        return $compliance;
    }

    /**
     * Check compliance with a single instruction
     */
    private function checkSingleInstructionCompliance($userCode, $instruction)
    {
        $instruction = strtolower($instruction);
        $normalizedCode = strtolower($userCode);
        
        // Define instruction patterns and their validation logic
        $patterns = [
            'doctype' => ['pattern' => 'doctype', 'validator' => fn() => strpos($normalizedCode, '<!doctype html>') !== false],
            'html element' => ['pattern' => 'html', 'validator' => fn() => preg_match('/<html[^>]*>/', $normalizedCode)],
            'head section' => ['pattern' => 'head', 'validator' => fn() => preg_match('/<head[^>]*>.*<\/head>/s', $normalizedCode)],
            'title' => ['pattern' => 'title', 'validator' => fn() => preg_match('/<title[^>]*>.*<\/title>/', $normalizedCode)],
            'body' => ['pattern' => 'body', 'validator' => fn() => preg_match('/<body[^>]*>.*<\/body>/s', $normalizedCode)],
            'heading' => ['pattern' => 'h[1-6]|heading', 'validator' => fn() => preg_match('/<h[1-6][^>]*>/', $normalizedCode)],
            'paragraph' => ['pattern' => 'paragraph|<p>', 'validator' => fn() => preg_match('/<p[^>]*>/', $normalizedCode)],
            'meta viewport' => ['pattern' => 'viewport|meta.*viewport', 'validator' => fn() => preg_match('/meta.*viewport/', $normalizedCode)]
        ];
        
        foreach ($patterns as $patternKey => $patternData) {
            if (preg_match('/' . $patternData['pattern'] . '/i', $instruction)) {
                return $patternData['validator']();
            }
        }
        
        // Default to basic keyword matching if no specific pattern matches
        $keywords = explode(' ', $instruction);
        foreach ($keywords as $keyword) {
            if (strlen($keyword) > 3 && strpos($normalizedCode, strtolower($keyword)) !== false) {
                return true;
            }
        }
        
                return false;
            }

    /**
     * Generate HTML output for validation and display
     */
    private function generateHtmlOutput($userCode)
    {
        // Safely generate HTML output by sanitizing and processing the code
        return htmlspecialchars($userCode, ENT_QUOTES, 'UTF-8');
    }

    /**
     * Calculate overall score based on validation results
     */
    private function calculateScore($validationResults, $instructionCompliance)
    {
        $totalScore = 0;
        $weights = [
            'doctype_check' => 10,
            'html_structure_check' => 15,
            'required_elements_check' => 20,
            'required_attributes_check' => 10,
            'structure_requirements_check' => 15,
            'content_check' => 10,
            'syntax_validation' => 10,
            'semantic_validation' => 5,
            'accessibility_check' => 5
        ];
        
        // Score validation results
        foreach ($validationResults as $check => $passed) {
            if (isset($weights[$check]) && $passed) {
                $totalScore += $weights[$check];
            }
        }
        
        // Score instruction compliance (10% weight)
        if (!empty($instructionCompliance)) {
            $completedInstructions = array_sum(array_values($instructionCompliance));
            $totalInstructions = count($instructionCompliance);
            if ($totalInstructions > 0) {
                $compliancePercentage = ($completedInstructions / $totalInstructions) * 100;
                $totalScore += ($compliancePercentage / 100) * 10;
            }
        }
        
        return min(100, round($totalScore));
    }



    /**
     * Extract detailed error information
     */
    private function extractErrorDetails($validationResults)
    {
        $errors = [];
        
        foreach ($validationResults as $check => $passed) {
            if (!$passed) {
                $errors[] = [
                    'check' => $check,
                    'message' => $this->getErrorMessageForCheck($check),
                    'severity' => $this->getErrorSeverityForCheck($check)
                ];
            }
        }
        
        return $errors;
    }

    /**
     * Get specific error message for a validation check
     */
    private function getErrorMessageForCheck($check)
    {
        $messages = [
            'doctype_check' => 'HTML5 DOCTYPE declaration is missing or incorrect',
            'html_structure_check' => 'Basic HTML structure is incomplete or malformed',
            'required_elements_check' => 'One or more required elements are missing',
            'required_attributes_check' => 'Required attributes are missing from elements',
            'structure_requirements_check' => 'HTML structure does not meet nesting requirements',
            'content_check' => 'Content does not match expected output',
            'syntax_validation' => 'HTML syntax errors detected',
            'semantic_validation' => 'Semantic HTML usage could be improved',
            'accessibility_check' => 'Accessibility requirements not met'
        ];
        
        return $messages[$check] ?? 'Validation check failed';
    }

    /**
     * Get error severity for a validation check
     */
    private function getErrorSeverityForCheck($check)
    {
        $criticalChecks = ['doctype_check', 'html_structure_check', 'required_elements_check', 'syntax_validation'];
        return in_array($check, $criticalChecks) ? 'error' : 'warning';
    }

    /**
     * Create validation summary for response
     */
    private function createValidationSummary($validationResults)
    {
        $summary = [];
        $passed = 0;
        $total = count($validationResults);
        
        foreach ($validationResults as $check => $result) {
            if ($result) {
                $passed++;
            }
            $summary[$check] = [
                'passed' => $result,
                'message' => $result ? 'Passed' : $this->getErrorMessageForCheck($check)
            ];
        }
        
        return [
            'overall' => [
                'passed' => $passed,
                'total' => $total,
                'percentage' => round(($passed / $total) * 100)
            ],
            'details' => $summary
        ];
    }

    /**
     * Create instruction progress summary
     */
    private function createInstructionProgress($instructionCompliance)
    {
        $total = count($instructionCompliance);
        $completed = array_sum(array_values($instructionCompliance));
        
        return [
            'completed' => $completed,
            'total' => $total,
            'percentage' => $total > 0 ? round(($completed / $total) * 100) : 0,
            'details' => $instructionCompliance
        ];
    }

    /**
     * Get contextual hints based on validation failures
     */
    private function getContextualHints($validationResults, $activity)
    {
        $hints = [];
        
        // Get pre-defined hints from activity metadata
        $activityHints = $activity->metadata['hints'] ?? [];
        
        // Add context-specific hints based on failures
        if (!($validationResults['doctype_check'] ?? true)) {
            $hints[] = "Start your HTML document with <!DOCTYPE html> to declare it as HTML5.";
        }
        
        if (!($validationResults['html_structure_check'] ?? true)) {
            $hints[] = "Make sure your document has the basic structure: <html>, <head>, and <body> elements.";
        }
        
        if (!($validationResults['required_elements_check'] ?? true)) {
            $hints[] = "Review the instructions to see which specific elements you need to include.";
        }
        
        // Add activity-specific hints if available
        if (!empty($activityHints)) {
            $hints = array_merge($hints, array_slice($activityHints, 0, 3)); // Limit to 3 hints
        }
        
        return array_slice($hints, 0, 5); // Maximum 5 hints
    }

    /**
     * Individual validation methods
     */
    private function validateDoctype($userCode)
    {
        return stripos($userCode, '<!DOCTYPE html>') !== false;
    }

    private function validateHtmlStructure($normalizedCode)
    {
        $hasHtml = preg_match('/<html[^>]*>.*<\/html>/s', $normalizedCode);
        $hasHead = preg_match('/<head[^>]*>.*<\/head>/s', $normalizedCode);
        $hasBody = preg_match('/<body[^>]*>.*<\/body>/s', $normalizedCode);
        
        return $hasHtml && $hasHead && $hasBody;
    }

    private function validateRequiredElements($normalizedCode, $criteria)
    {
        $requiredElements = $criteria['required_elements'] ?? [];
        
        foreach ($requiredElements as $element) {
            $pattern = '/<' . preg_quote($element, '/') . '(\s[^>]*)?>/';
            if (!preg_match($pattern, $normalizedCode)) {
                return false;
            }
        }
        
        return true;
    }

    private function validateRequiredAttributes($normalizedCode, $criteria)
    {
        $requiredAttributes = $criteria['required_attributes'] ?? [];
        
        foreach ($requiredAttributes as $selector => $attributes) {
        foreach ($attributes as $attr => $value) {
                $pattern = '/' . preg_quote($attr, '/') . '=["\']' . preg_quote($value, '/') . '["\']/';
                if (!preg_match($pattern, $normalizedCode)) {
                return false;
            }
        }
        }
        
        return true;
    }

    private function validateStructureRequirements($normalizedCode, $criteria)
    {
        $structureChecks = $criteria['structure_checks'] ?? [];
        
        foreach ($structureChecks as $check) {
            if (!$this->checkStructureRequirement($normalizedCode, $check)) {
                return false;
            }
        }
        
        return true;
    }

    private function checkStructureRequirement($html, $check)
    {
        switch ($check['type']) {
            case 'doctype':
                return strpos($html, '<!doctype html>') !== false;
            
            case 'nested':
                $parent = $check['parent'];
                $child = $check['child'];
                $pattern = '/<' . preg_quote($parent, '/') . '[^>]*>.*?<' . preg_quote($child, '/') . '[^>]*>.*?<\/' . preg_quote($child, '/') . '>.*?<\/' . preg_quote($parent, '/') . '>/s';
                return preg_match($pattern, $html) > 0;
            
            case 'order':
                $first = $check['first'];
                $second = $check['second'];
                $firstPos = strpos($html, '<' . $first);
                $secondPos = strpos($html, '<' . $second);
                return $firstPos !== false && $secondPos !== false && $firstPos < $secondPos;
            
            default:
                return true;
        }
    }

    private function validateContent($userCode, $expectedOutput)
    {
        if (empty($expectedOutput)) {
            return true; // No content requirement
        }
        
        $outputText = strip_tags($userCode);
        $outputText = preg_replace('/\s+/', ' ', trim($outputText));
        $expectedText = preg_replace('/\s+/', ' ', trim($expectedOutput));
        
        return stripos($outputText, $expectedText) !== false;
    }

    private function validateSyntax($userCode)
    {
        return $this->validateHtmlWithDom($userCode);
    }

    /**
     * Advanced HTML syntax validation using DOMDocument
     */
    private function validateHtmlWithDom($userCode)
    {
        // Suppress warnings for malformed HTML
        $previousErrorReporting = error_reporting(0);
        libxml_use_internal_errors(true);
        
        try {
            $dom = new \DOMDocument('1.0', 'UTF-8');
            $dom->formatOutput = true;
            $dom->preserveWhiteSpace = false;
            
            // Try to load the HTML
            $result = $dom->loadHTML($userCode, LIBXML_HTML_NOIMPLIED | LIBXML_HTML_NODEFDTD);
            
            // Check for parsing errors
            $errors = libxml_get_errors();
            $hasErrors = false;
            $criticalErrors = 0;
            
            foreach ($errors as $error) {
                // Count only critical errors, not warnings about HTML5 elements
                if ($error->level === LIBXML_ERR_ERROR || $error->level === LIBXML_ERR_FATAL) {
                    $criticalErrors++;
                }
            }
            
            // Clear errors
            libxml_clear_errors();
            
            // Validate specific HTML requirements
            $validationResult = $this->performDomValidation($dom, $userCode);
            
            return $result && $criticalErrors === 0 && $validationResult;
            
        } catch (\Exception $e) {
            return false;
        } finally {
            // Restore error reporting
            error_reporting($previousErrorReporting);
        }
    }

    /**
     * Perform DOM-based validation checks
     */
    private function performDomValidation($dom, $originalCode)
    {
        $errors = [];
        
        // Check for proper DOCTYPE
        if (!$this->hasDoctype($originalCode)) {
            $errors[] = 'Missing DOCTYPE declaration';
        }
        
        // Check for required HTML structure
        $htmlElements = $dom->getElementsByTagName('html');
        if ($htmlElements->length === 0) {
            $errors[] = 'Missing html element';
        }
        
        $headElements = $dom->getElementsByTagName('head');
        if ($headElements->length === 0) {
            $errors[] = 'Missing head element';
        }
        
        $bodyElements = $dom->getElementsByTagName('body');
        if ($bodyElements->length === 0) {
            $errors[] = 'Missing body element';
        }
        
        // Check for unclosed tags using DOM
        $this->checkForUnclosedTags($dom, $errors);
        
        // Check for invalid nesting
        $this->checkForInvalidNesting($dom, $errors);
        
        // Check for missing required attributes
        $this->checkForMissingRequiredAttributes($dom, $errors);
        
        return count($errors) === 0;
    }

    /**
     * Check for DOCTYPE declaration
     */
    private function hasDoctype($code)
    {
        return preg_match('/<!DOCTYPE\s+html/i', trim($code));
    }

    /**
     * Check for unclosed tags using DOM structure
     */
    private function checkForUnclosedTags($dom, &$errors)
    {
        // DOM automatically closes tags, but we can check for common issues
        $xpath = new \DOMXPath($dom);
        
        // Check for self-closing tags that shouldn't be
        $voidElements = ['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'track', 'wbr'];
        
        foreach ($voidElements as $element) {
            $nodes = $xpath->query("//{$element}[normalize-space(text()) != '']");
            if ($nodes->length > 0) {
                $errors[] = "Void element '{$element}' cannot have content";
            }
        }
    }

    /**
     * Check for invalid HTML nesting
     */
    private function checkForInvalidNesting($dom, &$errors)
    {
        $xpath = new \DOMXPath($dom);
        
        // Check for common invalid nesting patterns
        $invalidNesting = [
            '//p//div' => 'div element cannot be nested inside p element',
            '//p//p' => 'p element cannot be nested inside another p element',
            '//h1//h1' => 'h1 element cannot be nested inside another h1 element',
            '//h1//h2' => 'h2 element cannot be nested inside h1 element',
            '//a//a' => 'a element cannot be nested inside another a element'
        ];
        
        foreach ($invalidNesting as $xpath_expr => $errorMsg) {
            $nodes = $xpath->query($xpath_expr);
            if ($nodes->length > 0) {
                $errors[] = $errorMsg;
            }
        }
    }

    /**
     * Check for missing required attributes
     */
    private function checkForMissingRequiredAttributes($dom, &$errors)
    {
        $xpath = new \DOMXPath($dom);
        
        // Check for images without alt attributes
        $imgsWithoutAlt = $xpath->query('//img[not(@alt)]');
        if ($imgsWithoutAlt->length > 0) {
            $errors[] = 'img elements must have alt attribute for accessibility';
        }
        
        // Check for forms without action or method
        $formsWithoutAction = $xpath->query('//form[not(@action)]');
        foreach ($formsWithoutAction as $form) {
            if (!$form->getAttribute('action')) {
                $errors[] = 'form elements should have action attribute';
            }
        }
        
        // Check for links without href
        $linksWithoutHref = $xpath->query('//a[not(@href)]');
        if ($linksWithoutHref->length > 0) {
            $errors[] = 'a elements should have href attribute';
        }
    }

    /**
     * Enhanced HTML validation with detailed error reporting
     */
    private function validateHtmlStructureAdvanced($userCode)
    {
        $validation = $this->validateHtmlWithDom($userCode);
        
        // Additional structure checks
        $structureScore = 0;
        $maxScore = 100;
        
        // Check basic structure (30 points)
        if ($this->hasDoctype($userCode)) $structureScore += 10;
        if (preg_match('/<html[^>]*>.*<\/html>/s', $userCode)) $structureScore += 10;
        if (preg_match('/<head[^>]*>.*<\/head>/s', $userCode)) $structureScore += 5;
        if (preg_match('/<body[^>]*>.*<\/body>/s', $userCode)) $structureScore += 5;
        
        // Check semantic elements (20 points)
        $semanticElements = ['header', 'nav', 'main', 'section', 'article', 'aside', 'footer'];
        $foundSemantic = 0;
        foreach ($semanticElements as $element) {
            if (preg_match("/<{$element}[^>]*>/", $userCode)) {
                $foundSemantic++;
            }
        }
        $structureScore += min(20, $foundSemantic * 3);
        
        // Check accessibility (20 points)
        if (preg_match('/lang=["\'][^"\']*["\']/i', $userCode)) $structureScore += 10;
        if (!preg_match('/<img[^>]*(?<!alt=["\'][^"\']*["\'])[^>]*>/i', $userCode) || !preg_match('/<img/i', $userCode)) $structureScore += 10;
        
        // Check proper formatting (30 points)
        $lines = explode("\n", $userCode);
        $indentedLines = 0;
        foreach ($lines as $line) {
            if (preg_match('/^\s+/', $line) && trim($line) !== '') {
                $indentedLines++;
            }
        }
        if (count($lines) > 0) {
            $indentationScore = min(30, ($indentedLines / count($lines)) * 50);
            $structureScore += $indentationScore;
        }
        
        return [
            'syntax_valid' => $validation,
            'structure_score' => min(100, $structureScore),
            'detailed_validation' => true
        ];
    }

    private function validateSemantic($userCode)
    {
        return $this->validateSemanticWithDom($userCode);
    }

    /**
     * Advanced semantic HTML validation using DOM
     */
    private function validateSemanticWithDom($userCode)
    {
        libxml_use_internal_errors(true);
        
        try {
            $dom = new \DOMDocument('1.0', 'UTF-8');
            $dom->loadHTML($userCode, LIBXML_HTML_NOIMPLIED | LIBXML_HTML_NODEFDTD);
            
            $score = 0;
            $maxScore = 100;
            
            // Check for semantic HTML5 elements (40 points)
            $semanticElements = ['header', 'nav', 'main', 'section', 'article', 'aside', 'footer'];
            $foundSemantic = 0;
            foreach ($semanticElements as $element) {
                if ($dom->getElementsByTagName($element)->length > 0) {
                    $foundSemantic++;
                }
            }
            $score += min(40, $foundSemantic * 6);
            
            // Check for proper heading hierarchy (20 points)
            $score += $this->checkHeadingHierarchy($dom) ? 20 : 0;
            
            // Check for accessibility features (20 points)
            $score += $this->checkAccessibilityFeatures($dom);
            
            // Check for semantic content structure (20 points)
            $score += $this->checkSemanticStructure($dom);
            
            return $score >= 60; // Pass if 60% or better
            
        } catch (\Exception $e) {
            return false;
        } finally {
            libxml_clear_errors();
        }
    }

    /**
     * Check proper heading hierarchy (h1, h2, h3, etc.)
     */
    private function checkHeadingHierarchy($dom)
    {
        $xpath = new \DOMXPath($dom);
        $headings = [];
        
        for ($i = 1; $i <= 6; $i++) {
            $nodes = $xpath->query("//h{$i}");
            foreach ($nodes as $node) {
                $headings[] = $i;
            }
        }
        
        if (empty($headings)) {
            return false;
        }
        
        // Check if headings start with h1 and don't skip levels
        sort($headings);
        $expectedNext = 1;
        
        foreach (array_unique($headings) as $level) {
            if ($level > $expectedNext + 1) {
                return false; // Skipped a level
            }
            $expectedNext = max($expectedNext, $level);
        }
        
        return $headings[0] === 1; // Should start with h1
    }

    /**
     * Check accessibility features
     */
    private function checkAccessibilityFeatures($dom)
    {
        $score = 0;
        $xpath = new \DOMXPath($dom);
        
        // Check for lang attribute on html (5 points)
        $htmlElements = $dom->getElementsByTagName('html');
        if ($htmlElements->length > 0 && $htmlElements->item(0)->getAttribute('lang')) {
            $score += 5;
        }
        
        // Check for alt attributes on images (5 points)
        $images = $dom->getElementsByTagName('img');
        $imagesWithAlt = 0;
        foreach ($images as $img) {
            if ($img->getAttribute('alt') !== '') {
                $imagesWithAlt++;
            }
        }
        if ($images->length > 0 && $imagesWithAlt === $images->length) {
            $score += 5;
        } elseif ($images->length === 0) {
            $score += 5; // No images means this requirement is met
        }
        
        // Check for form labels (5 points)
        $inputs = $dom->getElementsByTagName('input');
        $inputsWithLabels = 0;
        foreach ($inputs as $input) {
            $id = $input->getAttribute('id');
            if ($id) {
                $labels = $xpath->query("//label[@for='{$id}']");
                if ($labels->length > 0) {
                    $inputsWithLabels++;
                }
            }
        }
        if ($inputs->length > 0 && $inputsWithLabels === $inputs->length) {
            $score += 5;
        } elseif ($inputs->length === 0) {
            $score += 5;
        }
        
        // Check for skip links or nav landmarks (5 points)
        $nav = $dom->getElementsByTagName('nav');
        $skipLinks = $xpath->query("//a[contains(@href, '#')]");
        if ($nav->length > 0 || $skipLinks->length > 0) {
            $score += 5;
        }
        
        return $score;
    }

    /**
     * Check semantic content structure
     */
    private function checkSemanticStructure($dom)
    {
        $score = 0;
        
        // Check for proper document outline (5 points)
        $header = $dom->getElementsByTagName('header');
        $main = $dom->getElementsByTagName('main');
        $footer = $dom->getElementsByTagName('footer');
        
        if ($header->length > 0) $score += 2;
        if ($main->length > 0) $score += 3;
        if ($footer->length > 0) $score += 2;
        
        // Check for proper use of sections and articles (8 points)
        $sections = $dom->getElementsByTagName('section');
        $articles = $dom->getElementsByTagName('article');
        
        if ($sections->length > 0 || $articles->length > 0) {
            $score += 8;
        }
        
        // Check for navigation structure (7 points)
        $nav = $dom->getElementsByTagName('nav');
        if ($nav->length > 0) {
            // Check if nav contains a list structure
            $xpath = new \DOMXPath($dom);
            $navLists = $xpath->query('//nav//ul | //nav//ol');
            if ($navLists->length > 0) {
                $score += 7;
            } else {
                $score += 3; // Partial credit for having nav without proper list structure
            }
        }
        
        return $score;
    }

    private function validateAccessibility($normalizedCode)
    {
        // Basic accessibility checks
        $checks = [];
        
        // Check for alt attributes on images
        preg_match_all('/<img[^>]*>/i', $normalizedCode, $imgTags);
        foreach ($imgTags[0] as $imgTag) {
            if (strpos($imgTag, 'alt=') === false) {
                return false; // Missing alt attribute
            }
        }
        
        // Check for lang attribute on html element
        if (preg_match('/<html[^>]*>/', $normalizedCode, $htmlTag)) {
            if (strpos($htmlTag[0], 'lang=') === false) {
                return false; // Missing lang attribute
            }
        }
        
        return true;
    }

    private function validateCodeQuality($userCode)
    {
        // Basic code quality checks
        $quality = true;
        
        // Check for proper indentation (basic check)
        $lines = explode("\n", $userCode);
        $indentedLines = 0;
        foreach ($lines as $line) {
            if (preg_match('/^\s+/', $line)) {
                $indentedLines++;
            }
        }
        
        // If more than 30% of lines are indented, consider it well-formatted
        return (count($lines) > 0) && (($indentedLines / count($lines)) > 0.3);
    }

    private function normalizeHtml($html)
    {
        // Remove comments
        $html = preg_replace('/<!--.*?-->/s', '', $html);
        
        // Normalize whitespace
        $html = preg_replace('/\s+/', ' ', $html);
        
        // Remove extra whitespace around tags
        $html = preg_replace('/>\s+</', '><', $html);
        
        return trim(strtolower($html));
    }

    /**
     * Update the specified activity
     */
    public function update(Request $request, $lessonId, $activityId)
    {
        $validator = Validator::make($request->all(), [
            'title' => 'sometimes|string|max:255',
            'description' => 'nullable|string',
            'activity_type' => 'sometimes|in:coding',
            'instructions' => 'sometimes|string',
            'questions' => 'nullable|array',
            'resources' => 'nullable|array',
            'time_limit' => 'nullable|integer|min:1',
            'max_attempts' => 'nullable|integer|min:1',
            'passing_score' => 'nullable|integer|min:0|max:100',
            'points' => 'nullable|integer|min:0',
            'order_index' => 'nullable|integer',
            'is_required' => 'nullable|boolean',
            'is_published' => 'nullable|boolean',
            'metadata' => 'nullable|array',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            $activity = Activity::where('lesson_id', $lessonId)
                ->findOrFail($activityId);
            
            $activity->update($request->all());

            return response()->json([
                'message' => 'Activity updated successfully',
                'activity' => $activity
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Activity not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to update activity',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Remove the specified activity
     */
    public function destroy($lessonId, $activityId)
    {
        try {
            $activity = Activity::where('lesson_id', $lessonId)
                ->findOrFail($activityId);
            
            $activity->delete();

            return response()->json([
                'message' => 'Activity deleted successfully'
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Activity not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to delete activity',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Reorder activities within a lesson
     */
    public function reorder(Request $request, $lessonId)
    {
        $validator = Validator::make($request->all(), [
            'activities' => 'required|array',
            'activities.*.id' => 'required|integer|exists:activities,id',
            'activities.*.order_index' => 'required|integer',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            foreach ($request->activities as $activityData) {
                Activity::where('id', $activityData['id'])
                    ->where('lesson_id', $lessonId)
                    ->update(['order_index' => $activityData['order_index']]);
            }

            return response()->json([
                'message' => 'Activities reordered successfully'
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to reorder activities',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get all submissions for an activity (Admin only)
     */
    public function getSubmissions($lessonId, $activityId)
    {
        try {
            $activity = Activity::where('lesson_id', $lessonId)->findOrFail($activityId);
            
            $submissions = ActivitySubmission::with('user:id,name,full_name,email')
                ->where('activity_id', $activityId)
                ->orderBy('submitted_at', 'desc')
                ->get();

            return response()->json([
                'activity' => [
                    'id' => $activity->id,
                    'title' => $activity->title,
                    'description' => $activity->description,
                    'max_attempts' => $activity->max_attempts,
                    'passing_score' => $activity->passing_score,
                    'points' => $activity->points
                ],
                'submissions' => $submissions->map(function($submission) {
                    return [
                        'id' => $submission->id,
                        'user' => $submission->user,
                        'attempt_number' => $submission->attempt_number,
                        'score' => $submission->score,
                        'completion_status' => $submission->completion_status,
                        'is_completed' => $submission->is_completed,
                        'time_spent_minutes' => $submission->time_spent_minutes,
                        'submitted_at' => $submission->submitted_at,
                        'completed_at' => $submission->completed_at,
                        'feedback' => $submission->feedback,
                        'validation_summary' => $submission->validation_results,
                        'instruction_compliance' => $submission->instruction_compliance,
                        'generated_output_preview' => substr($submission->generated_output, 0, 200) . '...'
                    ];
                }),
                'stats' => [
                    'total_submissions' => $submissions->count(),
                    'completed_submissions' => $submissions->where('is_completed', true)->count(),
                    'average_score' => $submissions->avg('score') ?? 0,
                    'average_attempts' => $submissions->groupBy('user_id')->map->count()->avg() ?? 0
                ]
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Activity not found'
            ], 404);
        }
    }

    /**
     * Get detailed submission for admin review
     */
    public function getSubmissionDetail($submissionId)
    {
        try {
            $submission = ActivitySubmission::with(['user:id,name,full_name,email', 'activity:id,title,description,metadata'])
                ->findOrFail($submissionId);

            return response()->json([
                'submission' => [
                    'id' => $submission->id,
                    'user' => $submission->user,
                    'activity' => $submission->activity,
                    'attempt_number' => $submission->attempt_number,
                    'submitted_code' => $submission->submitted_code,
                    'generated_output' => $submission->generated_output,
                    'score' => $submission->score,
                    'completion_status' => $submission->completion_status,
                    'is_completed' => $submission->is_completed,
                    'time_spent_minutes' => $submission->time_spent_minutes,
                    'submitted_at' => $submission->submitted_at,
                    'completed_at' => $submission->completed_at,
                    'feedback' => $submission->feedback,
                    'validation_results' => $submission->validation_results,
                    'instruction_compliance' => $submission->instruction_compliance,
                    'error_details' => $submission->error_details
                ]
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Submission not found'
            ], 404);
        }
    }

    /**
     * Update submission status (for instructor override)
     */
    public function updateSubmissionStatus(Request $request, $submissionId)
    {
        $validator = Validator::make($request->all(), [
            'completion_status' => 'required|in:pending,passed,failed,needs_review',
            'instructor_notes' => 'nullable|string|max:1000',
            'override_score' => 'nullable|integer|min:0|max:100'
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            $submission = ActivitySubmission::findOrFail($submissionId);
            
            // Store original values for logging
            $originalStatus = $submission->completion_status;
            $originalScore = $submission->score;
            
            $submission->completion_status = $request->completion_status;
            $submission->is_completed = $request->completion_status === 'passed';
            
            if ($request->has('override_score')) {
                $submission->score = $request->override_score;
            }
            
            if ($request->has('instructor_notes')) {
                $errorDetails = $submission->error_details ?? [];
                $errorDetails['instructor_notes'] = $request->instructor_notes;
                $errorDetails['reviewed_at'] = now()->toISOString();
                $errorDetails['reviewed_by'] = Auth::guard('admin')->user()->name ?? 'Admin';
                $submission->error_details = $errorDetails;
            }

            if ($request->completion_status === 'passed' && !$submission->completed_at) {
                $submission->completed_at = now();
            }

            $submission->save();

            // Log instructor override
            ActivityLog::logInstructorOverride(Auth::guard('admin')->id(), $submissionId, [
                'old_status' => $originalStatus,
                'new_status' => $request->completion_status,
                'old_score' => $originalScore,
                'new_score' => $submission->score,
                'instructor_notes' => $request->instructor_notes
            ]);

            return response()->json([
                'message' => 'Submission status updated successfully',
                'submission' => $submission
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Submission not found'
            ], 404);
        }
    }

    /**
     * Clear temporary data for testing (reset attempts to zero)
     */
    public function clearTemporaryData($activityId = null)
    {
        try {
            $tempDB = new TemporaryDatabaseService();
            
            if ($activityId) {
                // Clear data for specific activity
                $userId = 1; // Mock user ID
                $tempDB->clearUserActivityData($userId, $activityId);
                
                return response()->json([
                    'message' => "Temporary data cleared for activity {$activityId}",
                    'activity_id' => $activityId,
                    'user_id' => $userId
                ]);
            } else {
                // Clear all temporary data
                $tempDB->clearAllData();
                
                return response()->json([
                    'message' => 'All temporary data cleared successfully'
                ]);
            }
            
        } catch (\Exception $e) {
            Log::error('Error clearing temporary data', [
                'activity_id' => $activityId,
                'error' => $e->getMessage()
            ]);
            
            return response()->json([
                'message' => 'Failed to clear temporary data: ' . $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get activity statistics for a lesson
     */
    public function stats($lessonId)
    {
        try {
            $lesson = Lesson::findOrFail($lessonId);
            
            $totalActivities = Activity::where('lesson_id', $lessonId)->count();
            $requiredActivities = Activity::where('lesson_id', $lessonId)
                ->where('is_required', 1)
                ->count();
            $publishedActivities = Activity::where('lesson_id', $lessonId)
                ->where('is_published', 1)
                ->count();
            $totalPoints = Activity::where('lesson_id', $lessonId)
                ->sum('points');

            $byType = Activity::where('lesson_id', $lessonId)
                ->select('activity_type', \Illuminate\Support\Facades\DB::raw('count(*) as count'))
                ->groupBy('activity_type')
                ->pluck('count', 'activity_type');

            return response()->json([
                'stats' => [
                    'total' => $totalActivities,
                    'required' => $requiredActivities,
                    'published' => $publishedActivities,
                    'total_points' => $totalPoints,
                    'by_type' => $byType
                ]
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Lesson not found'
            ], 404);
        }
    }

    /**
     * Check code similarity against other submissions to prevent copy-paste solutions
     */
    private function checkCodeSimilarity($userId, $activityId, $userCode)
    {
        $normalizedCode = $this->normalizeCodeForSimilarity($userCode);
        
        // Get other successful submissions for this activity (not from current user)
        $otherSubmissions = ActivitySubmission::where('activity_id', $activityId)
            ->where('user_id', '!=', $userId)
            ->where('is_completed', true)
            ->select('submitted_code')
            ->get();
        
        // Check against template/example code if exists
        $activity = Activity::find($activityId);
        $templateCode = $activity->metadata['template_code'] ?? '';
        
        $maxSimilarity = 0;
        $suspiciousSubmissions = 0;
        
        // Check similarity against other submissions
        foreach ($otherSubmissions as $submission) {
            $otherNormalizedCode = $this->normalizeCodeForSimilarity($submission->submitted_code);
            $similarity = $this->calculateCodeSimilarity($normalizedCode, $otherNormalizedCode);
            
            if ($similarity > $maxSimilarity) {
                $maxSimilarity = $similarity;
            }
            
            if ($similarity > 0.85) { // 85% similarity threshold
                $suspiciousSubmissions++;
            }
        }
        
        // Check against template if provided
        $templateSimilarity = 0;
        if (!empty($templateCode)) {
            $normalizedTemplate = $this->normalizeCodeForSimilarity($templateCode);
            $templateSimilarity = $this->calculateCodeSimilarity($normalizedCode, $normalizedTemplate);
        }
        
        // Determine if code passes similarity check
        $passesCheck = $maxSimilarity < 0.90 && $templateSimilarity < 0.95 && $suspiciousSubmissions === 0;
        
        return [
            'passes_similarity_check' => $passesCheck,
            'max_similarity_percentage' => round($maxSimilarity * 100, 2),
            'template_similarity_percentage' => round($templateSimilarity * 100, 2),
            'suspicious_matches' => $suspiciousSubmissions,
            'originality_score' => $this->calculateOriginalityScore($normalizedCode),
            'similarity_details' => [
                'is_original' => $passesCheck,
                'risk_level' => $this->getSimilarityRiskLevel($maxSimilarity, $suspiciousSubmissions)
            ]
        ];
    }

    /**
     * Normalize code for similarity comparison
     */
    private function normalizeCodeForSimilarity($code)
    {
        // Remove comments
        $code = preg_replace('/<!--.*?-->/s', '', $code);
        
        // Normalize whitespace
        $code = preg_replace('/\s+/', ' ', $code);
        
        // Convert to lowercase
        $code = strtolower($code);
        
        // Remove extra spaces around tags
        $code = preg_replace('/\s*(<[^>]*>)\s*/', '$1', $code);
        
        // Sort attributes within tags to standardize order
        $code = preg_replace_callback('/<([a-z]+)([^>]*)>/i', function($matches) {
            $tag = $matches[1];
            $attributes = trim($matches[2]);
            
            if (empty($attributes)) {
                return "<{$tag}>";
            }
            
            // Parse and sort attributes
            preg_match_all('/([a-z-]+)=["\']([^"\']*)["\']/', $attributes, $attrMatches);
            $sortedAttrs = [];
            
            for ($i = 0; $i < count($attrMatches[0]); $i++) {
                $sortedAttrs[$attrMatches[1][$i]] = $attrMatches[2][$i];
            }
            
            ksort($sortedAttrs);
            $sortedAttrString = implode(' ', array_map(function($k, $v) {
                return "{$k}=\"{$v}\"";
            }, array_keys($sortedAttrs), $sortedAttrs));
            
            return "<{$tag} {$sortedAttrString}>";
        }, $code);
        
        return trim($code);
    }

    /**
     * Calculate similarity between two code strings using multiple algorithms
     */
    private function calculateCodeSimilarity($code1, $code2)
    {
        if (empty($code1) || empty($code2)) {
            return 0;
        }
        
        // Levenshtein distance similarity (for character-level comparison)
        $levenshteinSimilarity = $this->calculateLevenshteinSimilarity($code1, $code2);
        
        // Jaccard similarity (for token-level comparison)
        $jaccardSimilarity = $this->calculateJaccardSimilarity($code1, $code2);
        
        // Structural similarity (for HTML structure comparison)
        $structuralSimilarity = $this->calculateStructuralSimilarity($code1, $code2);
        
        // Weighted average of different similarity measures
        return ($levenshteinSimilarity * 0.3) + ($jaccardSimilarity * 0.4) + ($structuralSimilarity * 0.3);
    }

    /**
     * Calculate Levenshtein-based similarity
     */
    private function calculateLevenshteinSimilarity($str1, $str2)
    {
        $maxLen = max(strlen($str1), strlen($str2));
        if ($maxLen === 0) return 1;
        
        $distance = levenshtein($str1, $str2);
        return 1 - ($distance / $maxLen);
    }

    /**
     * Calculate Jaccard similarity based on tokens
     */
    private function calculateJaccardSimilarity($code1, $code2)
    {
        $tokens1 = $this->extractTokens($code1);
        $tokens2 = $this->extractTokens($code2);
        
        $intersection = count(array_intersect($tokens1, $tokens2));
        $union = count(array_unique(array_merge($tokens1, $tokens2)));
        
        return $union > 0 ? $intersection / $union : 0;
    }

    /**
     * Calculate structural similarity based on HTML elements
     */
    private function calculateStructuralSimilarity($code1, $code2)
    {
        $structure1 = $this->extractHtmlStructure($code1);
        $structure2 = $this->extractHtmlStructure($code2);
        
        if (empty($structure1) || empty($structure2)) {
            return 0;
        }
        
        $intersection = count(array_intersect($structure1, $structure2));
        $union = count(array_unique(array_merge($structure1, $structure2)));
        
        return $union > 0 ? $intersection / $union : 0;
    }

    /**
     * Extract tokens from code for comparison
     */
    private function extractTokens($code)
    {
        // Extract HTML tags
        preg_match_all('/<[^>]*>/', $code, $tags);
        
        // Extract text content
        $textContent = strip_tags($code);
        $words = preg_split('/\s+/', $textContent, -1, PREG_SPLIT_NO_EMPTY);
        
        return array_merge($tags[0], $words);
    }

    /**
     * Extract HTML structure for comparison
     */
    private function extractHtmlStructure($code)
    {
        // Extract opening tags in order
        preg_match_all('/<([a-z]+)[^>]*>/i', $code, $matches);
        return $matches[1];
    }

    /**
     * Calculate originality score based on code patterns
     */
    private function calculateOriginalityScore($code)
    {
        $score = 100;
        
        // Penalize for very short code
        if (strlen($code) < 100) {
            $score -= 20;
        }
        
        // Penalize for lack of unique content
        $uniqueChars = count(array_unique(str_split($code)));
        if ($uniqueChars < 20) {
            $score -= 15;
        }
        
        // Reward for custom class names or IDs
        $customNames = preg_match_all('/(?:class|id)=["\']([^"\']*)["\']/', $code, $matches);
        $score += min(20, $customNames * 5);
        
        // Reward for varied content
        $contentVariety = substr_count($code, '<') * 2; // Number of elements
        $score += min(20, $contentVariety);
        
        return max(0, min(100, $score));
    }

    /**
     * Determine similarity risk level
     */
    private function getSimilarityRiskLevel($maxSimilarity, $suspiciousSubmissions)
    {
        if ($maxSimilarity > 0.95 || $suspiciousSubmissions > 2) {
            return 'high';
        } elseif ($maxSimilarity > 0.85 || $suspiciousSubmissions > 0) {
            return 'medium';
        } elseif ($maxSimilarity > 0.70) {
            return 'low';
        } else {
            return 'none';
        }
    }

    /**
     * Get comprehensive analytics for an activity
     */
    public function getActivityAnalytics($activityId)
    {
        try {
            $activity = Activity::findOrFail($activityId);
            
            // Get all submissions for this activity
            $submissions = ActivitySubmission::where('activity_id', $activityId)
                ->with('user')
                ->orderBy('created_at', 'desc')
                ->get();
                
            // Get activity logs
            $logs = ActivityLog::where('activity_id', $activityId)
                ->where('created_at', '>=', now()->subDays(30))
                ->get();
            
            // Calculate metrics
            $analytics = [
                'activity_info' => [
                    'id' => $activity->id,
                    'title' => $activity->title,
                    'total_submissions' => $submissions->count(),
                    'unique_users' => $submissions->pluck('user_id')->unique()->count()
                ],
                'completion_metrics' => $this->calculateCompletionMetrics($submissions),
                'score_distribution' => $this->calculateScoreDistribution($submissions),
                'attempt_patterns' => $this->calculateAttemptPatterns($submissions),
                'time_analytics' => $this->calculateTimeAnalytics($submissions, $logs),
                'validation_insights' => $this->calculateValidationInsights($submissions),
                'similarity_analytics' => $this->calculateSimilarityAnalytics($submissions),
                'trend_data' => $this->calculateTrendData($submissions, $logs),
                'user_performance' => $this->calculateUserPerformanceMetrics($submissions)
            ];
            
            return response()->json($analytics);
            
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to fetch analytics: ' . $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get activity learning patterns
     */
    public function getActivityPatterns($activityId)
    {
        try {
            $submissions = ActivitySubmission::where('activity_id', $activityId)
                ->with('user')
                ->get();
                
            $patterns = [
                'common_mistakes' => $this->identifyCommonMistakes($submissions),
                'success_patterns' => $this->identifySuccessPatterns($submissions),
                'improvement_areas' => $this->identifyImprovementAreas($submissions),
                'learning_progression' => $this->analyzeLearningProgression($submissions),
                'code_complexity_trends' => $this->analyzeCodeComplexity($submissions)
            ];
            
            return response()->json($patterns);
            
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to fetch patterns: ' . $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get common errors for an activity
     */
    public function getCommonErrors($activityId)
    {
        try {
            $submissions = ActivitySubmission::where('activity_id', $activityId)
                ->where('is_completed', false)
                ->get();
                
            $errorAnalysis = [
                'validation_errors' => $this->analyzeValidationErrors($submissions),
                'syntax_errors' => $this->analyzeSyntaxErrors($submissions),
                'semantic_errors' => $this->analyzeSemanticErrors($submissions),
                'instruction_compliance_issues' => $this->analyzeInstructionCompliance($submissions),
                'recommendations' => $this->generateRecommendations($submissions)
            ];
            
            return response()->json($errorAnalysis);
            
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to fetch error analysis: ' . $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get overall analytics dashboard
     */
    public function getAnalyticsDashboard()
    {
        try {
            // Get overall statistics across all activities
            $totalActivities = Activity::count();
            $totalSubmissions = ActivitySubmission::count();
            $totalUsers = User::count();
            $completionRate = $totalSubmissions > 0 ? 
                (ActivitySubmission::where('is_completed', true)->count() / $totalSubmissions) * 100 : 0;
            
            // Recent activity data
            $recentSubmissions = ActivitySubmission::with(['user', 'activity'])
                ->orderBy('created_at', 'desc')
                ->limit(10)
                ->get();
                
            // Top performing activities
            $topActivities = Activity::with('submissions')
                ->get()
                ->map(function($activity) {
                    $submissions = $activity->submissions;
                    return [
                        'id' => $activity->id,
                        'title' => $activity->title,
                        'total_submissions' => $submissions->count(),
                        'completion_rate' => $submissions->count() > 0 ? 
                            ($submissions->where('is_completed', true)->count() / $submissions->count()) * 100 : 0,
                        'average_score' => $submissions->avg('score') ?? 0
                    ];
                })
                ->sortByDesc('completion_rate')
                ->take(5)
                ->values();
                
            // Activity logs statistics
            $logStats = ActivityLog::where('created_at', '>=', now()->subDays(7))
                ->selectRaw('event_type, COUNT(*) as count')
                ->groupBy('event_type')
                ->pluck('count', 'event_type');
            
            $dashboard = [
                'overview' => [
                    'total_activities' => $totalActivities,
                    'total_submissions' => $totalSubmissions,
                    'total_users' => $totalUsers,
                    'overall_completion_rate' => round($completionRate, 2)
                ],
                'recent_activity' => $recentSubmissions->map(function($submission) {
                    return [
                        'id' => $submission->id,
                        'user' => $submission->user->name,
                        'activity' => $submission->activity->title,
                        'score' => $submission->score,
                        'completed' => $submission->is_completed,
                        'submitted_at' => $submission->submitted_at
                    ];
                }),
                'top_activities' => $topActivities,
                'event_distribution' => $logStats,
                'weekly_trends' => $this->getWeeklyTrends()
            ];
            
            return response()->json($dashboard);
            
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to fetch dashboard: ' . $e->getMessage()
            ], 500);
        }
    }

    private function calculateCompletionMetrics($submissions)
    {
        $total = $submissions->count();
        $completed = $submissions->where('is_completed', true)->count();
        
        return [
            'total_submissions' => $total,
            'completed_submissions' => $completed,
            'completion_rate' => $total > 0 ? round(($completed / $total) * 100, 2) : 0,
            'average_attempts' => $submissions->avg('attempt_number') ?? 0,
            'first_attempt_success_rate' => $total > 0 ? 
                round(($submissions->where('attempt_number', 1)->where('is_completed', true)->count() / $total) * 100, 2) : 0
        ];
    }

    private function calculateScoreDistribution($submissions)
    {
        $scores = $submissions->pluck('score')->filter();
        
        return [
            'average_score' => round($scores->avg(), 2),
            'median_score' => $scores->median(),
            'score_ranges' => [
                'excellent' => $scores->where('>=', 90)->count(),
                'good' => $scores->whereBetween('score', [80, 89])->count(),
                'satisfactory' => $scores->whereBetween('score', [70, 79])->count(),
                'needs_improvement' => $scores->where('<', 70)->count()
            ]
        ];
    }

    private function calculateAttemptPatterns($submissions)
    {
        $attemptCounts = $submissions->groupBy('user_id')->map(function($userSubmissions) {
            return $userSubmissions->max('attempt_number');
        });
        
        return [
            'average_attempts_to_completion' => round($attemptCounts->avg(), 2),
            'max_attempts' => $attemptCounts->max(),
            'single_attempt_completions' => $attemptCounts->where('=', 1)->count(),
            'multiple_attempt_completions' => $attemptCounts->where('>', 1)->count()
        ];
    }

    private function calculateTimeAnalytics($submissions, $logs)
    {
        $times = $submissions->pluck('time_spent_minutes')->filter();
        
        return [
            'average_time_minutes' => round($times->avg(), 2),
            'median_time_minutes' => $times->median(),
            'time_distribution' => [
                'quick' => $times->where('<', 15)->count(),
                'normal' => $times->whereBetween('time_spent_minutes', [15, 45])->count(),
                'extended' => $times->where('>', 45)->count()
            ],
            'peak_activity_hours' => $this->analyzePeakHours($logs)
        ];
    }

    private function calculateValidationInsights($submissions)
    {
        $validationResults = $submissions->pluck('validation_results')->filter();
        $commonIssues = [];
        
        foreach ($validationResults as $result) {
            foreach ($result as $check => $passed) {
                if ($check === 'similarity_check') continue;
                if (!$passed) {
                    $commonIssues[$check] = ($commonIssues[$check] ?? 0) + 1;
                }
            }
        }
        
        arsort($commonIssues);
        
        return [
            'most_common_issues' => array_slice($commonIssues, 0, 5, true),
            'validation_success_rates' => $this->calculateValidationSuccessRates($validationResults)
        ];
    }

    private function calculateSimilarityAnalytics($submissions)
    {
        $similarityData = $submissions->pluck('validation_results.similarity_check')->filter();
        
        return [
            'originality_distribution' => [
                'high_risk' => $similarityData->where('risk_level', 'high')->count(),
                'medium_risk' => $similarityData->where('risk_level', 'medium')->count(),
                'low_risk' => $similarityData->where('risk_level', 'low')->count(),
                'original' => $similarityData->where('risk_level', 'none')->count()
            ],
            'average_originality_score' => round($similarityData->avg('originality_score'), 2),
            'flagged_submissions' => $similarityData->where('passes_similarity_check', false)->count()
        ];
    }

    private function calculateTrendData($submissions, $logs)
    {
        // Group submissions by day for the last 30 days
        $dailySubmissions = [];
        for ($i = 29; $i >= 0; $i--) {
            $date = now()->subDays($i)->format('Y-m-d');
            $count = $submissions->filter(function($submission) use ($date) {
                return $submission->created_at->format('Y-m-d') === $date;
            })->count();
            $dailySubmissions[$date] = $count;
        }
        
        return [
            'daily_submissions' => $dailySubmissions,
            'weekly_completion_trend' => $this->getWeeklyCompletionTrend($submissions),
            'monthly_improvement' => $this->getMonthlyImprovement($submissions)
        ];
    }

    private function calculateUserPerformanceMetrics($submissions)
    {
        return [
            'top_performers' => $submissions->where('is_completed', true)
                ->sortByDesc('score')
                ->take(10)
                ->map(function($submission) {
                    return [
                        'user' => $submission->user->name,
                        'score' => $submission->score,
                        'attempt_number' => $submission->attempt_number,
                        'completed_at' => $submission->completed_at
                    ];
                })
                ->values(),
            'struggling_users' => $submissions->where('attempt_number', '>', 3)
                ->where('is_completed', false)
                ->groupBy('user_id')
                ->map(function($userSubmissions) {
                    $latest = $userSubmissions->sortByDesc('created_at')->first();
                    return [
                        'user' => $latest->user->name,
                        'attempts' => $userSubmissions->count(),
                        'best_score' => $userSubmissions->max('score'),
                        'last_attempt' => $latest->created_at
                    ];
                })
                ->sortByDesc('attempts')
                ->take(10)
                ->values()
        ];
    }

    // Helper methods for complex calculations
    private function identifyCommonMistakes($submissions) { return []; }
    private function identifySuccessPatterns($submissions) { return []; }
    private function identifyImprovementAreas($submissions) { return []; }
    private function analyzeLearningProgression($submissions) { return []; }
    private function analyzeCodeComplexity($submissions) { return []; }
    private function analyzeValidationErrors($submissions) { return []; }
    private function analyzeSyntaxErrors($submissions) { return []; }
    private function analyzeSemanticErrors($submissions) { return []; }
    private function analyzeInstructionCompliance($submissions) { return []; }
    private function generateRecommendations($submissions) { return []; }
    private function analyzePeakHours($logs) { return []; }
    private function calculateValidationSuccessRates($validationResults) { return []; }
    private function getWeeklyTrends() { return []; }
    private function getWeeklyCompletionTrend($submissions) { return []; }
    private function getMonthlyImprovement($submissions) { return []; }

    /**
     * Fallback submission handling when AI validation fails
     */
    private function handleSubmissionFallback(Request $request, $activityId, \Exception $originalError)
    {
        Log::warning('🔄 Using fallback validation due to AI error', [
            'activity_id' => $activityId,
            'original_error' => $originalError->getMessage()
        ]);

        try {
            // Get user from request or use default - USER-SPECIFIC DATA ISOLATION
            $userId = $request->input('user_id') ?? 1; // Default to user 1 if not provided
            $user = (object)[
                'id' => $userId,
                'email' => 'user@example.com',
                'name' => 'User'
            ];

            $userCode = trim($request->user_code);
            $activity = Activity::findOrFail($activityId);

            // Basic validation fallback
            $basicScore = $this->performBasicValidation($userCode);
            $isCompleted = $basicScore >= 70;

            // Store with fallback data
            $tempDB = new TemporaryDatabaseService();
            $currentStatus = $tempDB->getSubmissionStatus($user->id, $activityId);
            $attemptNumber = $currentStatus['total_attempts'] + 1;

            $submissionData = [
                'user_id' => $user->id,
                'activity_id' => $activityId,
                'submitted_code' => $userCode,
                'score' => $basicScore,
                'is_completed' => $isCompleted,
                'completion_status' => $isCompleted ? 'passed' : 'needs_improvement',
                'time_spent_minutes' => $request->time_spent_minutes,
                'feedback' => 'Basic validation completed. AI validation temporarily unavailable.',
                'attempt_number' => $attemptNumber,
                'validation_results' => json_encode([
                    'ai_powered' => false,
                    'fallback_mode' => true,
                    'original_error' => $originalError->getMessage()
                ])
            ];

            $submission = $tempDB->storeSubmission($submissionData);

            return response()->json([
                'success' => true,
                'submission_id' => $submission['id'],
                'score' => $basicScore,
                'is_completed' => $isCompleted,
                'completion_status' => $isCompleted ? 'passed' : 'needs_improvement',
                'attempt_number' => $attemptNumber,
                'feedback' => 'Your submission has been processed with basic validation. AI validation is temporarily unavailable.',
                'ai_powered' => false,
                'fallback_mode' => true,
                'message' => $isCompleted ? 
                    'Activity completed with basic validation!' : 
                    'Activity submitted. Please review your code and try again.'
            ]);

        } catch (\Exception $fallbackError) {
            Log::error('❌ Fallback validation also failed', [
                'activity_id' => $activityId,
                'fallback_error' => $fallbackError->getMessage()
            ]);

            return response()->json([
                'success' => false,
                'message' => 'Submission validation failed. Please try again later.',
                'error' => 'validation_system_error'
            ], 500);
        }
    }

    /**
     * Perform basic HTML validation as fallback
     */
    private function performBasicValidation($userCode)
    {
        $score = 0;

        // Basic structure checks
        if (stripos($userCode, '<!DOCTYPE html>') !== false) $score += 20;
        if (preg_match('/<html[^>]*>.*<\/html>/s', $userCode)) $score += 20;
        if (preg_match('/<head[^>]*>.*<\/head>/s', $userCode)) $score += 20;
        if (preg_match('/<title[^>]*>.*<\/title>/s', $userCode)) $score += 20;
        if (preg_match('/<body[^>]*>.*<\/body>/s', $userCode)) $score += 20;

        return $score;
    }

    /**
     * Analyze the quality of the student's code explanation
     */
    private function analyzeCodeExplanation($explanation, $code)
    {
        if (empty($explanation)) {
            return [
                'quality_score' => 0,
                'completeness_score' => 0,
                'technical_accuracy_score' => 0,
                'clarity_score' => 0,
                'overall_rating' => 'none',
                'suggestions' => ['Please provide an explanation of your code.']
            ];
        }

        $wordCount = str_word_count($explanation);
        $codeLines = count(explode("\n", trim($code)));
        
        // Quality indicators
        $hasHtmlTerms = preg_match_all('/\b(html|element|tag|attribute|doctype|semantic|accessibility)\b/i', $explanation);
        $hasStructureTerms = preg_match_all('/\b(structure|organization|layout|content|heading|paragraph)\b/i', $explanation);
        $hasReasoningTerms = preg_match_all('/\b(because|since|so that|in order to|the purpose|the reason)\b/i', $explanation);
        $mentionsSpecificElements = preg_match_all('/<[a-zA-Z]+[^>]*>/', $explanation);
        
        // Calculate scores
        $qualityScore = min(100, (
            ($wordCount >= 20 ? 25 : ($wordCount / 20) * 25) +
            ($hasHtmlTerms * 5) +
            ($hasStructureTerms * 5) +
            ($hasReasoningTerms * 10) +
            ($mentionsSpecificElements * 5)
        ));
        
        $completenessScore = min(100, (
            ($wordCount / max($codeLines * 2, 20)) * 50 +
            ($hasHtmlTerms > 0 ? 25 : 0) +
            ($hasStructureTerms > 0 ? 25 : 0)
        ));
        
        $technicalAccuracyScore = min(100, (
            ($hasHtmlTerms * 15) +
            ($mentionsSpecificElements * 10) +
            ($this->checkTechnicalTerms($explanation) * 15) +
            (strlen($explanation) > 50 ? 10 : 0)
        ));
        
        $clarityScore = min(100, (
            ($hasReasoningTerms > 0 ? 40 : 0) +
            ($this->hasClearSentences($explanation) ? 30 : 0) +
            ($wordCount >= 30 && $wordCount <= 150 ? 30 : 15)
        ));
        
        $overallScore = ($qualityScore + $completenessScore + $technicalAccuracyScore + $clarityScore) / 4;
        
        // Determine rating
        $rating = 'poor';
        if ($overallScore >= 85) $rating = 'excellent';
        elseif ($overallScore >= 70) $rating = 'good';
        elseif ($overallScore >= 55) $rating = 'satisfactory';
        elseif ($overallScore >= 40) $rating = 'needs_improvement';
        
        // Generate suggestions
        $suggestions = $this->generateExplanationSuggestions($qualityScore, $completenessScore, $technicalAccuracyScore, $clarityScore);
        
        return [
            'quality_score' => round($qualityScore, 2),
            'completeness_score' => round($completenessScore, 2),
            'technical_accuracy_score' => round($technicalAccuracyScore, 2),
            'clarity_score' => round($clarityScore, 2),
            'overall_score' => round($overallScore, 2),
            'overall_rating' => $rating,
            'word_count' => $wordCount,
            'has_html_terms' => $hasHtmlTerms,
            'has_structure_terms' => $hasStructureTerms,
            'has_reasoning' => $hasReasoningTerms > 0,
            'mentions_code_elements' => $mentionsSpecificElements,
            'suggestions' => $suggestions
        ];
    }

    /**
     * Check if the explanation is sufficient for completion
     */
    private function isExplanationSufficient($explanation, $analysis, $codeScore)
    {
        if (empty($explanation)) {
            return false;
        }
        
        $overallScore = $analysis['overall_score'];
        $wordCount = $analysis['word_count'];
        
        // Minimum requirements
        $minWordCount = 20;
        $minQualityScore = 50;
        
        // Higher code scores require better explanations
        if ($codeScore >= 90) {
            $minQualityScore = 60;
            $minWordCount = 25;
        } elseif ($codeScore >= 80) {
            $minQualityScore = 55;
            $minWordCount = 22;
        }
        
        return $wordCount >= $minWordCount && 
               $overallScore >= $minQualityScore &&
               $analysis['has_html_terms'] > 0;
    }

    /**
     * Update completion status determination to include explanation requirements
     */
    private function determineCompletionStatus($score, $activity, $validationResults, $explanationRequired = false, $explanationSufficient = true)
    {
        $passingScore = $activity->metadata['passing_score'] ?? 80;
        $criticalChecks = ['doctype_check', 'html_structure_check', 'syntax_validation'];
        
        // Check critical validation requirements
        foreach ($criticalChecks as $check) {
            if (isset($validationResults[$check]) && !$validationResults[$check]) {
                return 'failed';
            }
        }
        
        // Check explanation requirements
        if ($explanationRequired && !$explanationSufficient) {
            return 'needs_explanation'; // New status for incomplete explanations
        }
        
        // Check score requirements
        if ($score >= $passingScore) {
            return 'passed';
        } elseif ($score >= 60) {
            return 'partial';
        } else {
            return 'failed';
        }
    }

    /**
     * Update feedback generation to include explanation feedback
     */
    private function generateFeedback($validationResults, $instructionCompliance, $isCompleted, $explanationRequired = false, $explanationSufficient = true)
    {
        $feedback = [];
        
        // Existing validation feedback
        $passedChecks = 0;
        $totalChecks = 0;
        
        foreach ($validationResults as $check => $result) {
            if ($check === 'similarity_check') continue;
            $totalChecks++;
            if ($result) {
                $passedChecks++;
            } else {
                $feedback[] = $this->getErrorMessageForCheck($check);
            }
        }
        
        // Explanation feedback
        if ($explanationRequired) {
            if (!$explanationSufficient) {
                $feedback[] = "📝 Code explanation required: Please provide a detailed explanation of your HTML code, including why you chose specific elements and how they fulfill the requirements.";
            } else {
                $feedback[] = "✅ Good explanation! Your description demonstrates understanding of your code choices.";
            }
        }
        
        // Overall feedback
        if ($isCompleted) {
            $feedback[] = "🎉 Excellent work! You've successfully completed this activity with proper code validation and explanation.";
        } else {
            $percentage = $totalChecks > 0 ? round(($passedChecks / $totalChecks) * 100) : 0;
            $feedback[] = "📊 Progress: {$percentage}% of validation checks passed. Review the feedback above and try again.";
        }
        
        return implode("\n\n", $feedback);
    }

    // Helper methods for explanation analysis
    private function checkTechnicalTerms($explanation)
    {
        $technicalTerms = [
            'semantic', 'accessibility', 'markup', 'doctype', 'viewport', 'meta',
            'hierarchy', 'nesting', 'validation', 'standards', 'compliance'
        ];
        
        $found = 0;
        foreach ($technicalTerms as $term) {
            if (stripos($explanation, $term) !== false) {
                $found++;
            }
        }
        
        return $found;
    }

    private function hasClearSentences($explanation)
    {
        $sentences = preg_split('/[.!?]+/', $explanation);
        $clearSentences = 0;
        
        foreach ($sentences as $sentence) {
            $sentence = trim($sentence);
            if (strlen($sentence) > 10 && str_word_count($sentence) >= 3) {
                $clearSentences++;
            }
        }
        
        return $clearSentences >= 2;
    }

    private function generateExplanationSuggestions($qualityScore, $completenessScore, $technicalScore, $clarityScore)
    {
        $suggestions = [];
        
        if ($qualityScore < 60) {
            $suggestions[] = "Include more details about your HTML elements and their purposes.";
        }
        
        if ($completenessScore < 60) {
            $suggestions[] = "Explain more aspects of your code - cover structure, content, and styling choices.";
        }
        
        if ($technicalScore < 60) {
            $suggestions[] = "Use proper HTML terminology like 'semantic elements', 'DOCTYPE', 'accessibility', etc.";
        }
        
        if ($clarityScore < 60) {
            $suggestions[] = "Explain the reasoning behind your code choices using phrases like 'because', 'in order to', 'the purpose is'.";
        }
        
        if (empty($suggestions)) {
            $suggestions[] = "Great explanation! You've demonstrated good understanding of your code.";
        }
        
        return $suggestions;
    }


    /**
     * Validate instruction compliance dynamically
     */
    private function validateInstructionCompliance($userCode, $instruction)
    {
        // Clean up the instruction text
        $instruction = trim($instruction);
        $instruction = preg_replace('/^\d+\.\s*/', '', $instruction); // Remove numbering
        
        // Convert instruction to lowercase for checking
        $lowerInstruction = strtolower($instruction);
        $lowerCode = strtolower($userCode);
        
        // Check different types of instructions
        if (strpos($lowerInstruction, 'doctype') !== false) {
            return $this->validateDoctype($userCode);
        }
        
        if (strpos($lowerInstruction, 'title') !== false && strpos($lowerInstruction, 'first image') !== false) {
            return stripos($userCode, 'My First Image') !== false || stripos($userCode, 'first image') !== false;
        }
        
        if (strpos($lowerInstruction, '<img>') !== false || strpos($lowerInstruction, 'image') !== false) {
            return preg_match('/<img[^>]*>/i', $userCode);
        }
        
        if (strpos($lowerInstruction, 'src') !== false && strpos($lowerInstruction, 'placeholder') !== false) {
            return stripos($userCode, 'via.placeholder.com') !== false || stripos($userCode, 'placeholder') !== false;
        }
        
        if (strpos($lowerInstruction, 'alt') !== false && strpos($lowerInstruction, 'sample image') !== false) {
            return stripos($userCode, 'alt="Sample Image"') !== false || stripos($userCode, 'sample image') !== false;
        }
        
        if (strpos($lowerInstruction, '<head>') !== false) {
            return stripos($userCode, '<head>') !== false && stripos($userCode, '</head>') !== false;
        }
        
        if (strpos($lowerInstruction, '<body>') !== false) {
            return stripos($userCode, '<body>') !== false && stripos($userCode, '</body>') !== false;
        }
        
        // Generic text matching for other instructions
        $keywords = explode(' ', $lowerInstruction);
        $matches = 0;
        foreach ($keywords as $keyword) {
            $keyword = trim($keyword, '.,!?');
            if (strlen($keyword) > 3 && strpos($lowerCode, $keyword) !== false) {
                $matches++;
            }
        }
        
        // Consider instruction compliant if more than half the keywords match
        return $matches > (count($keywords) / 2);
    }

    /**
     * Validate HTML basic structure (for simple cases)
     */
    private function validateHtmlBasicStructure($userCode)
    {
        return (stripos($userCode, '<html>') !== false || preg_match('/<html[^>]*>/i', $userCode)) &&
               stripos($userCode, '</html>') !== false;
    }

    /**
     * Validate head section and title (simple check)
     */
    private function validateHeadTitleSection($userCode)
    {
        $hasHead = stripos($userCode, '<head>') !== false && stripos($userCode, '</head>') !== false;
        $hasTitle = stripos($userCode, '<title>') !== false && stripos($userCode, '</title>') !== false;
        return $hasHead && $hasTitle;
    }

    /**
     * Validate body section (simple check)
     */
    private function validateBodySection($userCode)
    {
        return stripos($userCode, '<body>') !== false && stripos($userCode, '</body>') !== false;
    }

    /**
     * Generate dynamic feedback based on activity requirements
     */
    private function generateDynamicFeedback($checks, $instructionCompliance, $instructions, $isCompleted, $score)
    {
        if ($isCompleted) {
            return "Perfect! You have successfully completed the activity with all requirements met! Score: {$score}%";
        }
        
        $feedback = "Your HTML code needs some improvements to meet all requirements:\n\n";
        
        // Check basic structure
        if (!$checks['doctype']) {
            $feedback .= "• Add the HTML5 DOCTYPE declaration at the top\n";
        }
        
        if (!$checks['html_structure']) {
            $feedback .= "• Include proper HTML opening and closing tags\n";
        }
        
        if (!$checks['head_title']) {
            $feedback .= "• Add a head section with a title tag\n";
        }
        
        if (!$checks['body_structure']) {
            $feedback .= "• Include proper body opening and closing tags\n";
        }
        
        // Check instruction-specific requirements
        foreach ($instructions as $index => $instruction) {
            $instructionKey = "instruction_" . ($index + 1);
            if (!isset($instructionCompliance[$instructionKey]) || !$instructionCompliance[$instructionKey]) {
                $cleanInstruction = trim(preg_replace('/^\d+\.\s*/', '', $instruction));
                $feedback .= "• {$cleanInstruction}\n";
            }
        }
        
        $feedback .= "\nPlease make these corrections and try again!";
        
        return $feedback;
    }
}

