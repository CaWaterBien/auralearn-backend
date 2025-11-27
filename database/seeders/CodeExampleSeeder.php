<?php

namespace Database\Seeders;

use Illuminate\Database\Seeder;
use App\Models\CodeExample;
use App\Models\Lesson;
use App\Models\Topic;

class CodeExampleSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        // Get the first HTML lesson and its topics
        $htmlLesson = Lesson::whereHas('course', function($q) {
            $q->where('slug', 'html5-tutorial');
        })->where('slug', 'html-introduction')->first();

        if ($htmlLesson) {
            // Get topics for this lesson
            $topics = $htmlLesson->topics;
            $firstTopic = $topics->first();
            $secondTopic = $topics->skip(1)->first();

            // Create lesson-level code example
            CodeExample::create([
                'lesson_id' => $htmlLesson->id,
                'topic_id' => null,
                'title' => 'Basic HTML Structure',
                'description' => 'A complete HTML document with all essential elements',
                'language' => 'html',
                'initial_code' => '<!DOCTYPE html>
<html>
<head>
  <title>My First Web Page</title>
</head>
<body>
  <h1>Welcome to HTML!</h1>
  <p>This is my first paragraph.</p>
  <p>HTML is <strong>easy to learn</strong> and <em>fun to use</em>!</p>
</body>
</html>',
                'solution_code' => null,
                'hints' => 'Remember to include the DOCTYPE declaration and proper HTML structure.',
                'is_interactive' => true,
                'show_preview' => true,
                'allow_reset' => true,
                'order_index' => 0
            ]);

            // Create topic-specific code examples
            if ($firstTopic) {
                CodeExample::create([
                    'lesson_id' => $htmlLesson->id,
                    'topic_id' => $firstTopic->id,
                    'title' => 'HTML Headings Example',
                    'description' => 'Practice using different heading levels',
                    'language' => 'html',
                    'initial_code' => '<h1>Main Heading</h1>
<h2>Subheading</h2>
<h3>Section Title</h3>
<p>Content goes here...</p>',
                    'solution_code' => '<h1>Main Heading</h1>
<h2>Subheading</h2>
<h3>Section Title</h3>
<h4>Subsection</h4>
<h5>Minor Heading</h5>
<h6>Smallest Heading</h6>
<p>Content goes here with all heading levels!</p>',
                    'hints' => 'HTML has 6 heading levels from h1 to h6',
                    'is_interactive' => true,
                    'show_preview' => true,
                    'allow_reset' => true,
                    'order_index' => 0
                ]);
            }

            if ($secondTopic) {
                CodeExample::create([
                    'lesson_id' => $htmlLesson->id,
                    'topic_id' => $secondTopic->id,
                    'title' => 'HTML Lists Practice',
                    'description' => 'Create ordered and unordered lists',
                    'language' => 'html',
                    'initial_code' => '<h2>Shopping List</h2>
<ul>
  <li>Apples</li>
  <li>Bananas</li>
  <li>Oranges</li>
</ul>

<h2>Recipe Steps</h2>
<!-- Add an ordered list here -->',
                    'solution_code' => '<h2>Shopping List</h2>
<ul>
  <li>Apples</li>
  <li>Bananas</li>
  <li>Oranges</li>
</ul>

<h2>Recipe Steps</h2>
<ol>
  <li>Preheat the oven</li>
  <li>Mix ingredients</li>
  <li>Bake for 30 minutes</li>
  <li>Let cool and serve</li>
</ol>',
                    'hints' => 'Use <ol> for ordered lists and <ul> for unordered lists',
                    'is_interactive' => true,
                    'show_preview' => true,
                    'allow_reset' => true,
                    'order_index' => 0
                ]);
            }
        }

        // Get CSS lesson if exists
        $cssLesson = Lesson::whereHas('course', function($q) {
            $q->where('slug', 'css3-tutorial');
        })->first();

        if ($cssLesson) {
            CodeExample::create([
                'lesson_id' => $cssLesson->id,
                'topic_id' => null,
                'title' => 'CSS Styling Basics',
                'description' => 'Apply basic CSS styles to HTML elements',
                'language' => 'html',
                'initial_code' => '<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      font-family: Arial, sans-serif;
    }
    h1 {
      color: blue;
    }
    .highlight {
      background-color: yellow;
    }
  </style>
</head>
<body>
  <h1>CSS Example</h1>
  <p>This is a normal paragraph.</p>
  <p class="highlight">This paragraph is highlighted!</p>
</body>
</html>',
                'solution_code' => null,
                'hints' => 'CSS can be embedded in the <style> tag or linked externally',
                'is_interactive' => true,
                'show_preview' => true,
                'allow_reset' => true,
                'order_index' => 0
            ]);
        }

        $this->command->info('Code examples seeded successfully!');
    }
}
