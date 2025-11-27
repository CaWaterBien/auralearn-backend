<?php

namespace App\Http\Controllers;

use App\Models\CodeExample;
use App\Models\Lesson;
use App\Models\Topic;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Validator;

class CodeExampleController extends Controller
{
    /**
     * Display a listing of code examples
     */
    public function index(Request $request)
    {
        $query = CodeExample::with(['lesson', 'topic']);

        if ($request->has('lesson_id')) {
            $query->where('lesson_id', $request->lesson_id);
        }

        if ($request->has('topic_id')) {
            $query->where('topic_id', $request->topic_id);
        }

        $examples = $query->orderBy('order_index')->get();

        return response()->json([
            'code_examples' => $examples
        ]);
    }

    /**
     * Store a newly created code example
     */
    public function store(Request $request)
    {
        $validator = Validator::make($request->all(), [
            'lesson_id' => 'nullable|exists:lessons,id',
            'topic_id' => 'nullable|exists:topics,id',
            'title' => 'required|string|max:255',
            'description' => 'nullable|string',
            'language' => 'required|string|in:html,css,javascript,php,python,java,cpp,sql',
            'initial_code' => 'required|string',
            'solution_code' => 'nullable|string',
            'hints' => 'nullable|string',
            'is_interactive' => 'nullable|boolean',
            'show_preview' => 'nullable|boolean',
            'allow_reset' => 'nullable|boolean',
            'test_cases' => 'nullable|array',
            'order_index' => 'nullable|integer',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        // Ensure at least one parent is specified
        if (!$request->has('lesson_id') && !$request->has('topic_id')) {
            return response()->json([
                'message' => 'Either lesson_id or topic_id must be provided'
            ], 422);
        }

        try {
            $codeExample = CodeExample::create($request->all());

            return response()->json([
                'message' => 'Code example created successfully',
                'code_example' => $codeExample
            ], 201);

        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to create code example',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Display the specified code example
     */
    public function show($id)
    {
        try {
            $codeExample = CodeExample::with(['lesson', 'topic'])->findOrFail($id);

            return response()->json([
                'code_example' => $codeExample
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Code example not found'
            ], 404);
        }
    }

    /**
     * Update the specified code example
     */
    public function update(Request $request, $id)
    {
        $validator = Validator::make($request->all(), [
            'title' => 'sometimes|string|max:255',
            'description' => 'nullable|string',
            'language' => 'sometimes|string|in:html,css,javascript,php,python,java,cpp,sql',
            'initial_code' => 'sometimes|string',
            'solution_code' => 'nullable|string',
            'hints' => 'nullable|string',
            'is_interactive' => 'nullable|boolean',
            'show_preview' => 'nullable|boolean',
            'allow_reset' => 'nullable|boolean',
            'test_cases' => 'nullable|array',
            'order_index' => 'nullable|integer',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            $codeExample = CodeExample::findOrFail($id);
            $codeExample->update($request->all());

            return response()->json([
                'message' => 'Code example updated successfully',
                'code_example' => $codeExample
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Code example not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to update code example',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Remove the specified code example
     */
    public function destroy($id)
    {
        try {
            $codeExample = CodeExample::findOrFail($id);
            $codeExample->delete();

            return response()->json([
                'message' => 'Code example deleted successfully'
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Code example not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to delete code example',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Duplicate a code example
     */
    public function duplicate($id)
    {
        try {
            $original = CodeExample::findOrFail($id);
            
            $duplicate = $original->replicate();
            $duplicate->title = $original->title . ' (Copy)';
            $duplicate->save();

            return response()->json([
                'message' => 'Code example duplicated successfully',
                'code_example' => $duplicate
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Code example not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to duplicate code example',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Validate user code against test cases
     */
    public function validate(Request $request, $id)
    {
        $validator = Validator::make($request->all(), [
            'user_code' => 'required|string',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            $codeExample = CodeExample::findOrFail($id);
            
            // Here you would implement the actual code validation logic
            // This is a simplified version
            $isCorrect = false;
            $feedback = '';
            
            if ($codeExample->solution_code) {
                // Simple comparison (in reality, you'd want more sophisticated validation)
                $userCode = preg_replace('/\s+/', '', $request->user_code);
                $solutionCode = preg_replace('/\s+/', '', $codeExample->solution_code);
                
                if ($userCode === $solutionCode) {
                    $isCorrect = true;
                    $feedback = 'Great job! Your code is correct.';
                } else {
                    $feedback = 'Not quite right. Check your code and try again.';
                }
            }

            return response()->json([
                'is_correct' => $isCorrect,
                'feedback' => $feedback,
                'hints' => !$isCorrect ? $codeExample->hints : null
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Code example not found'
            ], 404);
        }
    }
}
