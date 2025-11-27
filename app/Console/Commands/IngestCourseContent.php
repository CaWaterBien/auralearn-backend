<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Services\RagEmbeddingService;
use App\Models\Course;
use App\Models\Lesson;
use App\Models\Topic;
use App\Models\Activity;
use App\Models\CodeExample;

class IngestCourseContent extends Command
{
    protected $signature = 'rag:ingest-courses
                            {--course-id= : Specific course ID to ingest}
                            {--clear : Clear existing course documents}';

    protected $description = 'Ingest course content (lessons, topics, activities) into RAG system';

    private RagEmbeddingService $embeddingService;

    public function __construct(RagEmbeddingService $embeddingService)
    {
        parent::__construct();
        $this->embeddingService = $embeddingService;
    }

    public function handle(): int
    {
        $courseId = $this->option('course-id');
        $shouldClear = $this->option('clear');

        if ($shouldClear) {
            $this->info('Clearing existing course documents...');
            \App\Models\RagDocument::whereIn('document_type', ['course', 'lesson', 'topic', 'activity', 'code_example'])
                ->delete();
        }

        // Get courses to process
        $courses = $courseId 
            ? Course::where('id', $courseId)->get()
            : Course::where('is_published', true)->get();

        if ($courses->isEmpty()) {
            $this->error('No courses found to ingest.');
            return 1;
        }

        $totalDocuments = 0;

        foreach ($courses as $course) {
            $this->info("Processing course: {$course->title}");
            $totalDocuments += $this->ingestCourse($course);
        }

        $this->info("Ingestion complete! Total documents created: {$totalDocuments}");
        return 0;
    }

    private function ingestCourse(Course $course): int
    {
        $documentsCreated = 0;

        // Ingest course overview
        if ($course->description) {
            $content = "Course: {$course->title}\n\n{$course->description}";
            $documentsCreated += $this->embeddingService->ingestDocument(
                $content,
                "course_{$course->id}",
                'course',
                [
                    'course_id' => $course->id,
                    'course_title' => $course->title,
                    'content_type' => 'course_overview'
                ]
            );
        }

        // Ingest lessons
        $lessons = Lesson::where('course_id', $course->id)->get();
        foreach ($lessons as $lesson) {
            $documentsCreated += $this->ingestLesson($lesson);
        }

        return $documentsCreated;
    }

    private function ingestLesson(Lesson $lesson): int
    {
        $documentsCreated = 0;

        // Ingest lesson content
        if ($lesson->content) {
            $content = "Lesson: {$lesson->title}\n\n{$lesson->content}";
            if ($lesson->objectives) {
                $content .= "\n\nObjectives:\n{$lesson->objectives}";
            }

            $documentsCreated += $this->embeddingService->ingestDocument(
                $content,
                "lesson_{$lesson->id}",
                'lesson',
                [
                    'course_id' => $lesson->course_id,
                    'lesson_id' => $lesson->id,
                    'lesson_title' => $lesson->title,
                    'content_type' => 'lesson_content'
                ]
            );
        }

        // Ingest topics
        $topics = Topic::where('lesson_id', $lesson->id)->get();
        foreach ($topics as $topic) {
            $documentsCreated += $this->ingestTopic($topic);
        }

        // Ingest activities
        $activities = Activity::where('lesson_id', $lesson->id)->get();
        foreach ($activities as $activity) {
            $documentsCreated += $this->ingestActivity($activity);
        }

        return $documentsCreated;
    }

    private function ingestTopic(Topic $topic): int
    {
        if (!$topic->content) {
            return 0;
        }

        $content = "Topic: {$topic->title}\n\n{$topic->content}";

        return $this->embeddingService->ingestDocument(
            $content,
            "topic_{$topic->id}",
            'topic',
            [
                'topic_id' => $topic->id,
                'lesson_id' => $topic->lesson_id,
                'topic_title' => $topic->title,
                'content_type' => 'topic_content'
            ]
        );
    }

    private function ingestActivity(Activity $activity): int
    {
        $documentsCreated = 0;

        // Ingest activity instructions
        if ($activity->instructions) {
            $content = "Activity: {$activity->title}\n\nInstructions:\n{$activity->instructions}";
            
            if ($activity->expected_output) {
                $content .= "\n\nExpected Output:\n{$activity->expected_output}";
            }

            $documentsCreated += $this->embeddingService->ingestDocument(
                $content,
                "activity_{$activity->id}",
                'activity',
                [
                    'activity_id' => $activity->id,
                    'lesson_id' => $activity->lesson_id,
                    'activity_title' => $activity->title,
                    'activity_type' => $activity->type,
                    'content_type' => 'activity_instructions'
                ]
            );
        }

        // Ingest starter code if available
        if ($activity->starter_code) {
            $content = "Activity Starter Code for: {$activity->title}\n\n```html\n{$activity->starter_code}\n```";
            
            $documentsCreated += $this->embeddingService->ingestDocument(
                $content,
                "activity_starter_{$activity->id}",
                'code_example',
                [
                    'activity_id' => $activity->id,
                    'lesson_id' => $activity->lesson_id,
                    'activity_title' => $activity->title,
                    'content_type' => 'starter_code'
                ]
            );
        }

        return $documentsCreated;
    }
}

