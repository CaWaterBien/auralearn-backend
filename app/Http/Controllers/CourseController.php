<?php

namespace App\Http\Controllers;

use App\Models\Course;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Validator;
use Illuminate\Support\Str;

class CourseController extends Controller
{
    /**
     * Display a listing of courses
     */
    public function index(Request $request)
    {
        $query = Course::with(['lessons' => function($q) {
            $q->select('id', 'course_id', 'title', 'slug', 'order_index', 'is_locked');
        }]);

        // Filter by published status
        if ($request->has('published')) {
            $query->where('is_published', $request->boolean('published'));
        }

        // Filter by category
        if ($request->has('category')) {
            $query->where('category', $request->category);
        }

        // Search
        if ($request->has('search')) {
            $search = $request->search;
            $query->where(function($q) use ($search) {
                $q->where('title', 'LIKE', "%{$search}%")
                  ->orWhere('description', 'LIKE', "%{$search}%");
            });
        }

        $courses = $query->orderBy('order_index')->orderBy('created_at', 'desc')->get();

        return response()->json([
            'courses' => $courses->map(function($course) {
                return [
                    'id' => $course->id,
                    'title' => $course->title,
                    'slug' => $course->slug,
                    'description' => $course->description,
                    'category' => $course->category,
                    'difficulty_level' => $course->difficulty_level,
                    'total_lessons' => $course->total_lessons,
                    'duration_hours' => $course->duration_hours,
                    'tags' => $course->tags,
                    'thumbnail' => $course->thumbnail,
                    'is_free' => $course->is_free,
                    'is_published' => $course->is_published,
                    'order_index' => $course->order_index,
                    'lessons_count' => $course->lessons->count(),
                    'created_at' => $course->created_at->toISOString(),
                ];
            })
        ]);
    }

    /**
     * Store a newly created course
     */
    public function store(Request $request)
    {
        $validator = Validator::make($request->all(), [
            'title' => 'required|string|max:255',
            'description' => 'nullable|string',
            'category' => 'nullable|string|max:100',
            'difficulty_level' => 'nullable|in:beginner,intermediate,advanced',
            'tags' => 'nullable|array',
            'thumbnail' => 'nullable|string',
            'is_free' => 'nullable|boolean',
            'is_published' => 'nullable|boolean',
            'order_index' => 'nullable|integer',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            $data = $request->all();
            $data['slug'] = Str::slug($request->title);
            
            // Ensure unique slug
            $originalSlug = $data['slug'];
            $count = 1;
            while (Course::where('slug', $data['slug'])->exists()) {
                $data['slug'] = $originalSlug . '-' . $count;
                $count++;
            }

            $course = Course::create($data);

            return response()->json([
                'message' => 'Course created successfully',
                'course' => $course
            ], 201);

        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to create course',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Display the specified course
     */
    public function show($id)
    {
        try {
            $course = Course::with([
                'lessons' => function($q) {
                    $q->with(['topics', 'codeExamples'])->orderBy('order_index');
                }
            ])->findOrFail($id);

            return response()->json([
                'course' => $course
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Course not found'
            ], 404);
        }
    }

    /**
     * Update the specified course
     */
    public function update(Request $request, $id)
    {
        $validator = Validator::make($request->all(), [
            'title' => 'sometimes|string|max:255',
            'description' => 'nullable|string',
            'category' => 'nullable|string|max:100',
            'difficulty_level' => 'nullable|in:beginner,intermediate,advanced',
            'tags' => 'nullable|array',
            'thumbnail' => 'nullable|string',
            'is_free' => 'nullable|boolean',
            'is_published' => 'nullable|boolean',
            'order_index' => 'nullable|integer',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            $course = Course::findOrFail($id);
            
            $data = $request->all();
            
            // Update slug if title changed
            if ($request->has('title') && $request->title !== $course->title) {
                $data['slug'] = Str::slug($request->title);
                
                // Ensure unique slug
                $originalSlug = $data['slug'];
                $count = 1;
                while (Course::where('slug', $data['slug'])->where('id', '!=', $id)->exists()) {
                    $data['slug'] = $originalSlug . '-' . $count;
                    $count++;
                }
            }

            $course->update($data);

            return response()->json([
                'message' => 'Course updated successfully',
                'course' => $course
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Course not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to update course',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Remove the specified course
     */
    public function destroy($id)
    {
        try {
            $course = Course::findOrFail($id);
            $course->delete();

            return response()->json([
                'message' => 'Course deleted successfully'
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Course not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to delete course',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Toggle course published status
     */
    public function togglePublished($id)
    {
        try {
            $course = Course::findOrFail($id);
            $course->is_published = !$course->is_published;
            $course->save();

            return response()->json([
                'message' => 'Course status updated successfully',
                'course' => $course
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Course not found'
            ], 404);
        }
    }

    /**
     * Get course by slug (for public access)
     */
    public function getBySlug($slug)
    {
        try {
            $course = Course::with([
                'publishedLessons' => function($q) {
                    $q->with([
                        'topics' => function($tq) {
                            $tq->with('codeExamples')->orderBy('order_index');
                        }, 
                        'codeExamples' => function($cq) {
                            $cq->orderBy('order_index');
                        }, 
                        'activities' => function($aq) {
                            $aq->where('is_published', 1);
                        }
                    ])->orderBy('order_index');
                }
            ])->where('slug', $slug)
              ->where('is_published', 1)
              ->firstOrFail();

            // Transform data to match frontend expectations
            $transformedCourse = [
                'id' => $course->id,
                'title' => $course->title,
                'slug' => $course->slug,
                'description' => $course->description,
                'category' => $course->category,
                'difficulty_level' => $course->difficulty_level,
                'total_lessons' => $course->total_lessons,
                'duration_hours' => $course->duration_hours,
                'tags' => $course->tags,
                'thumbnail' => $course->thumbnail,
                'is_free' => $course->is_free,
                'is_published' => $course->is_published,
                'created_at' => $course->created_at->toISOString(),
                'lessons' => $course->publishedLessons->map(function($lesson) {
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
                        'lesson_type' => $lesson->lesson_type,
                        'objectives' => $lesson->objectives,
                        'topics' => $lesson->topics->map(function($topic) {
                            return [
                                'id' => $topic->id,
                                'lesson_id' => $topic->lesson_id,
                                'title' => $topic->title,
                                'content' => $topic->content,
                                'content_type' => $topic->content_type,
                                'order_index' => $topic->order_index,
                                'codeExamples' => $topic->codeExamples->map(function($example) {
                                    return [
                                        'id' => $example->id,
                                        'lesson_id' => $example->lesson_id,
                                        'topic_id' => $example->topic_id,
                                        'title' => $example->title,
                                        'description' => $example->description,
                                        'language' => $example->language,
                                        'initial_code' => $example->initial_code,
                                        'solution_code' => $example->solution_code,
                                        'hints' => $example->hints,
                                        'is_interactive' => $example->is_interactive,
                                        'show_preview' => $example->show_preview,
                                        'allow_reset' => $example->allow_reset,
                                        'order_index' => $example->order_index,
                                        'created_at' => $example->created_at->toISOString(),
                                    ];
                                }),
                                'created_at' => $topic->created_at->toISOString(),
                            ];
                        }),
                        'codeExamples' => $lesson->codeExamples->map(function($example) {
                            return [
                                'id' => $example->id,
                                'lesson_id' => $example->lesson_id,
                                'topic_id' => $example->topic_id,
                                'title' => $example->title,
                                'description' => $example->description,
                                'language' => $example->language,
                                'initial_code' => $example->initial_code,
                                'solution_code' => $example->solution_code,
                                'hints' => $example->hints,
                                'is_interactive' => $example->is_interactive,
                                'show_preview' => $example->show_preview,
                                'allow_reset' => $example->allow_reset,
                                'order_index' => $example->order_index,
                                'created_at' => $example->created_at->toISOString(),
                            ];
                        }),
                        'activities' => $lesson->activities->map(function($activity) {
                            return [
                                'id' => $activity->id,
                                'lesson_id' => $activity->lesson_id,
                                'title' => $activity->title,
                                'description' => $activity->description,
                                'activity_type' => $activity->activity_type,
                                'instructions' => $activity->instructions,
                                'questions' => $activity->questions,
                                'resources' => $activity->resources,
                                'time_limit' => $activity->time_limit,
                                'max_attempts' => $activity->max_attempts,
                                'passing_score' => $activity->passing_score,
                                'points' => $activity->points,
                                'is_required' => $activity->is_required,
                                'created_at' => $activity->created_at->toISOString(),
                            ];
                        }),
                        'created_at' => $lesson->created_at->toISOString(),
                    ];
                })
            ];

            return response()->json([
                'course' => $transformedCourse
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'Course not found'
            ], 404);
        }
    }
}
