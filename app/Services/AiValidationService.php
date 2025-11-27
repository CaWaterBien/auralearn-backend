<?php

namespace App\Services;

use Illuminate\Support\Facades\Log;

class AiValidationService
{
    private NebiusClient $nebiusClient;

    public function __construct(NebiusClient $nebiusClient)
    {
        $this->nebiusClient = $nebiusClient;
    }

    /**
     * AI-powered validation of user code against activity instructions
     */
    public function validateCodeWithAi($userCode, $instructions, $activityTitle, $activityDescription = null)
    {
        try {
            Log::info('Starting AI validation', [
                'activity_title' => $activityTitle,
                'code_length' => strlen($userCode),
                'instructions_count' => count($instructions)
            ]);

            // Prepare the AI prompt for validation
            $prompt = $this->buildValidationPrompt($userCode, $instructions, $activityTitle, $activityDescription);
            
            // Format prompt as messages array for NebiusClient
            $messages = [
                [
                    'role' => 'system',
                    'content' => 'You are an expert HTML instructor evaluating student code submissions. Provide detailed, structured validation in JSON format.'
                ],
                [
                    'role' => 'user',
                    'content' => $prompt
                ]
            ];
            
            // Get AI analysis
            $aiResponse = $this->nebiusClient->createChatCompletion($messages);
            
            // Parse the AI response into structured validation data
            $validationResult = $this->parseAiValidationResponse($aiResponse);
            
            Log::info('AI validation completed', [
                'overall_score' => $validationResult['overall_score'],
                'completion_status' => $validationResult['completion_status']
            ]);

            return $validationResult;

        } catch (\Exception $e) {
            Log::error('AI validation failed, falling back to basic validation', [
                'error' => $e->getMessage()
            ]);

            // Fallback to basic validation if AI fails
            return $this->getFallbackValidation($userCode, $instructions);
        }
    }

    /**
     * Build comprehensive validation prompt for AI
     */
    private function buildValidationPrompt($userCode, $instructions, $activityTitle, $activityDescription)
    {
        $instructionsText = is_array($instructions) ? implode("\n", $instructions) : $instructions;
        
        return "You are an expert HTML instructor evaluating a student's code submission. Please analyze the following HTML code against the specific requirements and provide detailed validation.

**ACTIVITY: {$activityTitle}**
" . ($activityDescription ? "**DESCRIPTION: {$activityDescription}**\n" : "") . "
**REQUIREMENTS:**
{$instructionsText}

**STUDENT'S HTML CODE:**
```html
{$userCode}
```

**EVALUATION INSTRUCTIONS:**
Please analyze this code and respond with a JSON object containing:

1. **overall_score** (0-100): Overall score based on requirement completion
2. **com    sed', 'partial', 'failed'): Whether student passes (≥80%), partial (60-79%), or fails (<60%)
3. **requirements_analysis**: Array of each requirement with:
   - requirement: The specific requirement text
   - met: true/false if requirement is fulfilled
   - score: 0-100 score for this requirement
   - explanation: Detailed explanation of why it passes/fails
4. **technical_validation**: Object with:
   - html_structure: true/false (proper DOCTYPE, html, head, body)
   - syntax_valid: true/false (no syntax errors)
   - semantic_quality: 0-100 (use of appropriate HTML elements)
   - accessibility: 0-100 (alt attributes, lang attribute, etc.)
5. **detailed_feedback**: Brief, focused feedback (max 200 characters) with only:
   - What is missing or incorrect
   - Quick fix suggestions
   - No lengthy explanations
6. **suggestions**: Array of max 3 brief, actionable fixes (short phrases only)
7. **positive_aspects**: Array of max 2 brief positives (not displayed in frontend)
8. **areas_for_improvement**: Array of max 3 specific issues to fix (short phrases only)

**SCORING CRITERIA:**
- Each requirement should be weighted equally
- Consider both technical correctness and requirement fulfillment
- Be strict but educational in your evaluation
- Provide constructive feedback that helps learning

**IMPORTANT:** 
- Be very specific but CONCISE in your analysis
- Look for exact matches to requirements, not just similar implementations
- Keep all feedback SHORT and ACTIONABLE
- Focus only on what needs to be fixed, not long explanations
- Use brief phrases, not full sentences in suggestions/improvements

Please respond ONLY with the JSON object, no additional text.";
    }

    /**
     * Parse AI response into structured validation result
     */
    private function parseAiValidationResponse($aiResponse)
    {
        try {
            // Extract content from AI response (it's nested in the response structure)
            if (is_array($aiResponse) && isset($aiResponse['choices'][0]['message']['content'])) {
                $responseContent = $aiResponse['choices'][0]['message']['content'];
            } elseif (is_array($aiResponse) && isset($aiResponse['content'])) {
                $responseContent = $aiResponse['content'];
            } else {
                $responseContent = is_string($aiResponse) ? $aiResponse : json_encode($aiResponse);
            }
            
            // Clean the response (remove any markdown code blocks or extra text)
            $cleanResponse = $this->cleanJsonResponse($responseContent);
            
            // Parse JSON response from AI
            $aiResult = json_decode($cleanResponse, true);
            
            if (json_last_error() !== JSON_ERROR_NONE) {
                throw new \Exception('Invalid JSON response from AI: ' . json_last_error_msg());
            }

            // Validate and structure the response
            return [
                'ai_powered' => true,
                'overall_score' => $this->validateScore($aiResult['overall_score'] ?? 0),
                'completion_status' => $this->validateCompletionStatus($aiResult['completion_status'] ?? 'failed'),
                'is_completed' => ($aiResult['overall_score'] ?? 0) >= 80,
                'requirements_analysis' => $aiResult['requirements_analysis'] ?? [],
                'technical_validation' => $this->validateTechnicalResults($aiResult['technical_validation'] ?? []),
                'detailed_feedback' => $aiResult['detailed_feedback'] ?? 'No detailed feedback available.',
                'suggestions' => $aiResult['suggestions'] ?? [],
                'positive_aspects' => $aiResult['positive_aspects'] ?? [],
                'areas_for_improvement' => $aiResult['areas_for_improvement'] ?? [],
                'validation_summary' => $this->createValidationSummary($aiResult),
                'instruction_progress' => $this->createInstructionProgress($aiResult['requirements_analysis'] ?? [])
            ];

        } catch (\Exception $e) {
            Log::error('Failed to parse AI validation response', [
                'error' => $e->getMessage(),
                'response_preview' => is_string($aiResponse) ? substr($aiResponse, 0, 200) : substr(json_encode($aiResponse), 0, 200)
            ]);

            throw $e;
        }
    }

    /**
     * Clean JSON response from potential markdown or extra text
     */
    private function cleanJsonResponse($response)
    {
        // Remove markdown code blocks
        $response = preg_replace('/```json\s*/', '', $response);
        $response = preg_replace('/```\s*$/', '', $response);
        
        // Find JSON object boundaries
        $start = strpos($response, '{');
        $end = strrpos($response, '}');
        
        if ($start !== false && $end !== false && $end > $start) {
            return substr($response, $start, $end - $start + 1);
        }
        
        return trim($response);
    }

    /**
     * Validate and normalize score
     */
    private function validateScore($score)
    {
        $score = floatval($score);
        return max(0, min(100, $score));
    }

    /**
     * Validate completion status
     */
    private function validateCompletionStatus($status)
    {
        $validStatuses = ['passed', 'partial', 'failed'];
        return in_array($status, $validStatuses) ? $status : 'failed';
    }

    /**
     * Validate technical validation results
     */
    private function validateTechnicalResults($technical)
    {
        return [
            'html_structure' => $technical['html_structure'] ?? false,
            'syntax_valid' => $technical['syntax_valid'] ?? false,
            'semantic_quality' => $this->validateScore($technical['semantic_quality'] ?? 0),
            'accessibility' => $this->validateScore($technical['accessibility'] ?? 0)
        ];
    }

    /**
     * Create validation summary from AI results
     */
    private function createValidationSummary($aiResult)
    {
        $technical = $aiResult['technical_validation'] ?? [];
        $requirements = $aiResult['requirements_analysis'] ?? [];
        
        $passedChecks = 0;
        $totalChecks = 0;

        // Count technical checks
        foreach (['html_structure', 'syntax_valid'] as $check) {
            $totalChecks++;
            if ($technical[$check] ?? false) {
                $passedChecks++;
            }
        }

        // Count requirement checks
        foreach ($requirements as $req) {
            $totalChecks++;
            if ($req['met'] ?? false) {
                $passedChecks++;
            }
        }

        return [
            'overall' => [
                'passed' => $passedChecks,
                'total' => $totalChecks,
                'percentage' => $totalChecks > 0 ? round(($passedChecks / $totalChecks) * 100) : 0
            ],
            'details' => [
                'html_structure_check' => [
                    'passed' => $technical['html_structure'] ?? false,
                    'message' => ($technical['html_structure'] ?? false) ? 
                        'HTML structure is valid' : 'HTML structure needs improvement'
                ],
                'syntax_validation' => [
                    'passed' => $technical['syntax_valid'] ?? false,
                    'message' => ($technical['syntax_valid'] ?? false) ? 
                        'Syntax is valid' : 'Syntax errors detected'
                ],
                'semantic_quality' => [
                    'passed' => ($technical['semantic_quality'] ?? 0) >= 70,
                    'message' => ($technical['semantic_quality'] ?? 0) >= 70 ? 
                        'Good semantic HTML usage' : 'Semantic HTML usage could be improved'
                ],
                'accessibility_check' => [
                    'passed' => ($technical['accessibility'] ?? 0) >= 70,
                    'message' => ($technical['accessibility'] ?? 0) >= 70 ? 
                        'Good accessibility practices' : 'Accessibility could be improved'
                ]
            ]
        ];
    }

    /**
     * Create instruction progress from requirements analysis
     */
    private function createInstructionProgress($requirements)
    {
        if (empty($requirements)) {
            return [
                'completed' => 0,
                'total' => 0,
                'percentage' => 0,
                'details' => []
            ];
        }

        $completed = 0;
        $details = [];

        foreach ($requirements as $index => $req) {
            $key = "requirement_" . ($index + 1);
            $met = $req['met'] ?? false;
            
            if ($met) {
                $completed++;
            }
            
            $details[$key] = $met;
        }

        return [
            'completed' => $completed,
            'total' => count($requirements),
            'percentage' => count($requirements) > 0 ? round(($completed / count($requirements)) * 100) : 0,
            'details' => $details
        ];
    }

    /**
     * Generate concise educational feedback
     */
    public function generateEducationalFeedback($validationResult)
    {
        $feedback = [];
        
        if ($validationResult['is_completed']) {
            return "🎉 **Perfect!** All requirements completed successfully.";
        }

        // Only show what needs to be fixed
        if (!empty($validationResult['areas_for_improvement'])) {
            $feedback[] = "🔧 **Fix these issues:**";
            foreach (array_slice($validationResult['areas_for_improvement'], 0, 3) as $area) {
                $feedback[] = "• " . $area;
            }
        }

        // Show only the top 2 specific suggestions
        if (!empty($validationResult['suggestions'])) {
            $feedback[] = "\n💡 **Quick fixes:**";
            foreach (array_slice($validationResult['suggestions'], 0, 2) as $suggestion) {
                $feedback[] = "• " . $suggestion;
            }
        }

        return implode("\n", $feedback);
    }

    /**
     * Fallback validation when AI is unavailable
     */
    private function getFallbackValidation($userCode, $instructions)
    {
        Log::info('Using fallback validation');

        // Basic checks
        $hasDoctype = stripos($userCode, '<!DOCTYPE html>') !== false;
        $hasHtml = preg_match('/<html[^>]*>.*<\/html>/s', $userCode);
        $hasHead = preg_match('/<head[^>]*>.*<\/head>/s', $userCode);
        $hasBody = preg_match('/<body[^>]*>.*<\/body>/s', $userCode);
        $hasTitle = preg_match('/<title[^>]*>.*<\/title>/s', $userCode);

        $basicScore = 0;
        if ($hasDoctype) $basicScore += 20;
        if ($hasHtml) $basicScore += 20;
        if ($hasHead) $basicScore += 20;
        if ($hasTitle) $basicScore += 20;
        if ($hasBody) $basicScore += 20;

        return [
            'ai_powered' => false,
            'overall_score' => $basicScore,
            'completion_status' => $basicScore >= 80 ? 'passed' : ($basicScore >= 60 ? 'partial' : 'failed'),
            'is_completed' => $basicScore >= 80,
            'technical_validation' => [
                'html_structure' => $hasHtml && $hasHead && $hasBody,
                'syntax_valid' => true, // Basic assumption
                'semantic_quality' => 50, // Default
                'accessibility' => 50 // Default
            ],
            'detailed_feedback' => 'Basic validation completed. AI validation was unavailable, so this is a simplified check.',
            'suggestions' => [
                'Ensure you have all required HTML elements',
                'Check that all tags are properly closed',
                'Follow the activity instructions carefully'
            ],
            'positive_aspects' => [],
            'areas_for_improvement' => []
        ];
    }

    /**
     * Quick AI feedback for failed submissions
     */
    public function generateQuickFeedback($userCode, $requirements, $score)
    {
        try {
            $prompt = "As an HTML instructor, provide brief, encouraging feedback for a student whose code scored {$score}%. 

Requirements they need to meet:
" . implode("\n", $requirements) . "

Their code:
```html
{$userCode}
```

Provide 2-3 sentences of constructive feedback focusing on what they can improve. Be encouraging and specific.";

            $messages = [
                [
                    'role' => 'system',
                    'content' => 'You are an HTML instructor providing brief, constructive feedback.'
                ],
                [
                    'role' => 'user',
                    'content' => $prompt
                ]
            ];
            
            return $this->nebiusClient->createChatCompletion($messages);

        } catch (\Exception $e) {
            Log::error('Quick feedback generation failed', ['error' => $e->getMessage()]);
            return "Keep working on meeting the requirements! Review the instructions carefully and ensure your HTML structure is complete.";
        }
    }
}
