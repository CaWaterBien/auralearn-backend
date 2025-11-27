<?php

namespace App\Http\Controllers;

use App\Models\Topic;
use App\Models\Lesson;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Validator;

class TopicController extends Controller
{
    /**
     * Display a listing of topics for a lesson
     */
    public function index($lessonId)
    {
        try {
            $lesson = Lesson::findOrFail($lessonId);
            $topics = $lesson->topics()
                ->with('codeExamples')
                ->orderBy('order_index')
                ->get();

            return response()->json([
                'topics' => $topics
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Lesson not found'
            ], 404);
        }
    }

    /**
     * Store a newly created topic
     */
    public function store(Request $request, $lessonId)
    {
        $validator = Validator::make($request->all(), [
            'title' => 'required|string|max:255',
            'content' => 'required|string',
            'content_type' => 'nullable|in:text,code,video,image',
            'order_index' => 'nullable|integer',
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

            $topic = Topic::create($data);

            return response()->json([
                'message' => 'Topic created successfully',
                'topic' => $topic
            ], 201);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Lesson not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to create topic',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Display the specified topic
     */
    public function show($lessonId, $topicId)
    {
        try {
            $topic = Topic::with('codeExamples')
                ->where('lesson_id', $lessonId)
                ->findOrFail($topicId);

            return response()->json([
                'topic' => $topic
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Topic not found'
            ], 404);
        }
    }

    /**
     * Update the specified topic
     */
    public function update(Request $request, $lessonId, $topicId)
    {
        $validator = Validator::make($request->all(), [
            'title' => 'sometimes|string|max:255',
            'content' => 'sometimes|string',
            'content_type' => 'nullable|in:text,code,video,image',
            'order_index' => 'nullable|integer',
            'metadata' => 'nullable|array',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            $topic = Topic::where('lesson_id', $lessonId)->findOrFail($topicId);
            $topic->update($request->all());

            return response()->json([
                'message' => 'Topic updated successfully',
                'topic' => $topic
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Topic not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to update topic',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Remove the specified topic
     */
    public function destroy($lessonId, $topicId)
    {
        try {
            $topic = Topic::where('lesson_id', $lessonId)->findOrFail($topicId);
            $topic->delete();

            return response()->json([
                'message' => 'Topic deleted successfully'
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Topic not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to delete topic',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Reorder topics within a lesson
     */
    public function reorder(Request $request, $lessonId)
    {
        $validator = Validator::make($request->all(), [
            'topics' => 'required|array',
            'topics.*.id' => 'required|integer|exists:topics,id',
            'topics.*.order_index' => 'required|integer',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            foreach ($request->topics as $topicData) {
                Topic::where('id', $topicData['id'])
                    ->where('lesson_id', $lessonId)
                    ->update(['order_index' => $topicData['order_index']]);
            }

            return response()->json([
                'message' => 'Topics reordered successfully'
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to reorder topics',
                'error' => $e->getMessage()
            ], 500);
        }
    }
}
