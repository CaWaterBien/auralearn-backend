<?php

namespace App\Http\Controllers;

use App\Models\Lesson;
use App\Models\Course;
use App\Models\UserProgress;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Validator;
use Illuminate\Support\Str;
use Illuminate\Support\Facades\Auth;

class LessonController extends Controller
{
    /**
     * Display a listing of lessons for a course
     */
    public function index($courseId)
    {
        try {
            $course = Course::findOrFail($courseId);
            $lessons = $course->lessons()
                ->withCount(['topics', 'codeExamples', 'activities'])
                ->with([
                    'topics', 
                    'codeExamples' => function($q) {
                        $q->orderBy('order_index');
                    }, 
                    'activities'
                ])
                ->orderBy('order_index')
                ->get();

            return response()->json([
                'lessons' => $lessons->map(function($lesson) {
                    return [
                        'id' => $lesson->id,
                        'course_id' => $lesson->course_id,
                        'title' => $lesson->title,
                        'slug' => $lesson->slug,
                        'description' => $lesson->description,
                        'content' => $lesson->content,
                        'order_index' => $lesson->order_index,
                        'duration_minutes' => $lesson->duration_minutes,
                        'is_locked' => $lesson->is_locked,
                        'is_published' => $lesson->is_published,
                        'lesson_type' => $lesson->lesson_type,
                        'objectives' => $lesson->objectives,
                        'prerequisites' => $lesson->prerequisites,
                        'topics' => $lesson->topics,
                        'codeExamples' => $lesson->codeExamples,
                        'activities' => $lesson->activities,
                        'topics_count' => $lesson->topics_count ?? $lesson->topics->count(),
                        'exercises_count' => $lesson->codeExamples_count ?? $lesson->codeExamples->count(),
                        'activities_count' => $lesson->activities_count ?? $lesson->activities->count(),
                        'created_at' => $lesson->created_at->toISOString(),
                        'updated_at' => $lesson->updated_at->toISOString(),
                    ];
                })
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Course not found'
            ], 404);
        }
    }

    /**
     * Store a newly created lesson
     */
    public function store(Request $request, $courseId)
    {
        $validator = Validator::make($request->all(), [
            'title' => 'required|string|max:255',
            'description' => 'nullable|string',
            'content' => 'nullable|string',
            'order_index' => 'nullable|integer',
            'duration_minutes' => 'nullable|integer',
            'is_locked' => 'nullable|boolean',
            'is_published' => 'nullable|boolean',
            'lesson_type' => 'nullable|in:text,video,quiz,interactive',
            'objectives' => 'nullable|array',
            'prerequisites' => 'nullable|array',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            $course = Course::findOrFail($courseId);
            
            $data = $request->all();
            $data['course_id'] = $courseId;
            $data['slug'] = Str::slug($request->title);
            
            // Ensure unique slug within course
            $originalSlug = $data['slug'];
            $count = 1;
            while (Lesson::where('course_id', $courseId)->where('slug', $data['slug'])->exists()) {
                $data['slug'] = $originalSlug . '-' . $count;
                $count++;
            }

            $lesson = Lesson::create($data);

            return response()->json([
                'message' => 'Lesson created successfully',
                'lesson' => $lesson
            ], 201);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Course not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to create lesson',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Display the specified lesson
     */
    public function show($courseId, $lessonId)
    {
        try {
            // Check if this is an admin request to load all content including unpublished
            $isAdminRequest = request()->is('*/admin/*') || str_contains(request()->path(), 'admin');
            
            $lesson = Lesson::with([
                'topics' => function($q) {
                    $q->with('codeExamples')->orderBy('order_index');
                }, 
                'codeExamples' => function($q) {
                    $q->orderBy('order_index');
                },
                'activities' => function($q) use ($isAdminRequest) {
                    // For admin, load all activities; for users, only published ones
                    if (!$isAdminRequest) {
                        $q->where('is_published', 1);
                    }
                    $q->orderBy('order_index');
                }
            ])
            ->where('course_id', $courseId)
            ->findOrFail($lessonId);

            // Get user progress if authenticated
            $userProgress = null;
            if (Auth::check()) {
                $userProgress = UserProgress::where('user_id', Auth::id())
                    ->where('course_id', $courseId)
                    ->where('lesson_id', $lessonId)
                    ->first();
            }

            return response()->json([
                'lesson' => [
                    'id' => $lesson->id,
                    'course_id' => $lesson->course_id,
                    'title' => $lesson->title,
                    'slug' => $lesson->slug,
                    'description' => $lesson->description,
                    'content' => $lesson->content,
                    'order_index' => $lesson->order_index,
                    'duration_minutes' => $lesson->duration_minutes,
                    'is_locked' => $lesson->is_locked,
                    'is_published' => $lesson->is_published,
                    'lesson_type' => $lesson->lesson_type,
                    'objectives' => $lesson->objectives,
                    'prerequisites' => $lesson->prerequisites,
                    'topics' => $lesson->topics,
                    'codeExamples' => $lesson->codeExamples,
                    'activities' => $lesson->activities,
                    'topics_count' => $lesson->topics->count(),
                    'exercises_count' => $lesson->codeExamples->count(),
                    'activities_count' => $lesson->activities->count(),
                    'created_at' => $lesson->created_at->toISOString(),
                    'updated_at' => $lesson->updated_at->toISOString(),
                ],
                'progress' => $userProgress
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Lesson not found'
            ], 404);
        }
    }

    /**
     * Update the specified lesson
     */
    public function update(Request $request, $courseId, $lessonId)
    {
        $validator = Validator::make($request->all(), [
            'title' => 'sometimes|string|max:255',
            'description' => 'nullable|string',
            'content' => 'nullable|string',
            'order_index' => 'nullable|integer',
            'duration_minutes' => 'nullable|integer',
            'is_locked' => 'nullable|boolean',
            'is_published' => 'nullable|boolean',
            'lesson_type' => 'nullable|in:text,video,quiz,interactive',
            'objectives' => 'nullable|array',
            'prerequisites' => 'nullable|array',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            $lesson = Lesson::where('course_id', $courseId)->findOrFail($lessonId);
            
            $data = $request->all();
            
            // Update slug if title changed
            if ($request->has('title') && $request->title !== $lesson->title) {
                $data['slug'] = Str::slug($request->title);
                
                // Ensure unique slug within course
                $originalSlug = $data['slug'];
                $count = 1;
                while (Lesson::where('course_id', $courseId)
                    ->where('slug', $data['slug'])
                    ->where('id', '!=', $lessonId)
                    ->exists()) {
                    $data['slug'] = $originalSlug . '-' . $count;
                    $count++;
                }
            }

            $lesson->update($data);

            return response()->json([
                'message' => 'Lesson updated successfully',
                'lesson' => $lesson
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Lesson not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to update lesson',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Remove the specified lesson
     */
    public function destroy($courseId, $lessonId)
    {
        try {
            $lesson = Lesson::where('course_id', $courseId)->findOrFail($lessonId);
            $lesson->delete();

            return response()->json([
                'message' => 'Lesson deleted successfully'
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Lesson not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to delete lesson',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Reorder lessons within a course
     */
    public function reorder(Request $request, $courseId)
    {
        $validator = Validator::make($request->all(), [
            'lessons' => 'required|array',
            'lessons.*.id' => 'required|integer|exists:lessons,id',
            'lessons.*.order_index' => 'required|integer',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            foreach ($request->lessons as $lessonData) {
                Lesson::where('id', $lessonData['id'])
                    ->where('course_id', $courseId)
                    ->update(['order_index' => $lessonData['order_index']]);
            }

            return response()->json([
                'message' => 'Lessons reordered successfully'
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to reorder lessons',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Mark lesson as completed for authenticated user
     */
    public function markComplete(Request $request, $courseId, $lessonId)
    {
        if (!Auth::check()) {
            return response()->json(['message' => 'Unauthorized'], 401);
        }

        try {
            $lesson = Lesson::where('course_id', $courseId)->findOrFail($lessonId);
            
            $progress = UserProgress::firstOrCreate(
                [
                    'user_id' => Auth::id(),
                    'course_id' => $courseId,
                    'lesson_id' => $lessonId,
                ],
                [
                    'started_at' => now(),
                ]
            );

            $progress->is_completed = true;
            $progress->completed_at = now();
            $progress->completion_percentage = 100;
            $progress->save();

            return response()->json([
                'message' => 'Lesson marked as complete',
                'progress' => $progress
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Lesson not found'
            ], 404);
        }
    }
}
