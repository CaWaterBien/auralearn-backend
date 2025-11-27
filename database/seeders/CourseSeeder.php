<?php

namespace Database\Seeders;

use Illuminate\Database\Seeder;
use App\Models\Course;
use App\Models\Lesson;
use App\Models\Topic;
use App\Models\CodeExample;

class CourseSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        // Create HTML5 Tutorial Course
        $htmlCourse = Course::create([
            'title' => 'HTML5 Tutorial',
            'slug' => 'html5-tutorial',
            'description' => 'HTML is the standard markup language for Web pages. Learn HTML5 from scratch with this comprehensive tutorial.',
            'category' => 'web-development',
            'difficulty_level' => 'beginner',
            'tags' => ['HTML', 'HTML5', 'Web Development', 'Frontend'],
            'is_free' => true,
            'is_published' => true,
            'order_index' => 1,
        ]);

        // Create lessons for HTML course
        $lesson1 = Lesson::create([
            'course_id' => $htmlCourse->id,
            'title' => 'HTML Introduction',
            'slug' => 'html-introduction',
            'description' => 'Learn what HTML is and why it\'s important for web development.',
            'content' => '<h2>What is HTML?</h2><p>HTML stands for Hyper Text Markup Language. It is the standard markup language for creating Web pages.</p><p>HTML describes the structure of a Web page and consists of a series of elements that tell the browser how to display the content.</p>',
            'lesson_type' => 'text',
            'duration_minutes' => 15,
            'is_published' => true,
            'order_index' => 1,
        ]);

        // Add topics to lesson 1
        Topic::create([
            'lesson_id' => $lesson1->id,
            'title' => 'What is HTML?',
            'content' => 'HTML (HyperText Markup Language) is the most basic building block of the Web. It defines the meaning and structure of web content. Other technologies besides HTML are generally used to describe a web page\'s appearance/presentation (CSS) or functionality/behavior (JavaScript).',
            'content_type' => 'text',
            'order_index' => 1,
        ]);

        Topic::create([
            'lesson_id' => $lesson1->id,
            'title' => 'HTML Elements',
            'content' => 'HTML elements are the building blocks of HTML pages. An HTML element is defined by a start tag, some content, and an end tag. For example: <tagname>Content goes here...</tagname>',
            'content_type' => 'text',
            'order_index' => 2,
        ]);

        // Add a code example to lesson 1
        CodeExample::create([
            'lesson_id' => $lesson1->id,
            'title' => 'Try it Yourself - Your First HTML Page',
            'description' => 'Create your first HTML page with basic structure',
            'language' => 'html',
            'initial_code' => '<!DOCTYPE html>
<html>
<head>
    <title>Page Title</title>
</head>
<body>
    <h1>My First Heading</h1>
    <p>My first paragraph.</p>
</body>
</html>',
            'solution_code' => '<!DOCTYPE html>
<html>
<head>
    <title>My First Web Page</title>
</head>
<body>
    <h1>Welcome to HTML</h1>
    <p>This is my first web page!</p>
</body>
</html>',
            'hints' => 'Try changing the text inside the <h1> and <p> tags to customize your page.',
            'is_interactive' => true,
            'show_preview' => true,
            'allow_reset' => true,
            'order_index' => 1,
        ]);

        $lesson2 = Lesson::create([
            'course_id' => $htmlCourse->id,
            'title' => 'HTML Editors',
            'slug' => 'html-editors',
            'description' => 'Learn about different HTML editors and how to set up your development environment.',
            'content' => '<h2>HTML Editors</h2><p>Web pages can be created and modified by using professional HTML editors. However, for learning HTML we recommend a simple text editor like Notepad (PC) or TextEdit (Mac).</p>',
            'lesson_type' => 'text',
            'duration_minutes' => 10,
            'is_published' => true,
            'order_index' => 2,
        ]);

        $lesson3 = Lesson::create([
            'course_id' => $htmlCourse->id,
            'title' => 'HTML Basic',
            'slug' => 'html-basic',
            'description' => 'Learn the basic HTML tags and document structure.',
            'content' => '<h2>HTML Documents</h2><p>All HTML documents must start with a document type declaration: &lt;!DOCTYPE html&gt;. The HTML document itself begins with &lt;html&gt; and ends with &lt;/html&gt;.</p>',
            'lesson_type' => 'interactive',
            'duration_minutes' => 20,
            'is_locked' => true,
            'is_published' => true,
            'order_index' => 3,
        ]);

        // Update course lesson count and duration
        $htmlCourse->updateLessonCount();

        // Create CSS3 Tutorial Course
        $cssCourse = Course::create([
            'title' => 'CSS3 Complete Guide',
            'slug' => 'css3-complete-guide',
            'description' => 'Master CSS3 and learn how to style modern web pages with animations, flexbox, and grid.',
            'category' => 'web-development',
            'difficulty_level' => 'intermediate',
            'tags' => ['CSS', 'CSS3', 'Styling', 'Web Design'],
            'is_free' => true,
            'is_published' => true,
            'order_index' => 2,
        ]);

        // Create JavaScript Course
        $jsCourse = Course::create([
            'title' => 'JavaScript Programming',
            'slug' => 'javascript-programming',
            'description' => 'Learn JavaScript from basics to advanced concepts including ES6+, async programming, and more.',
            'category' => 'programming',
            'difficulty_level' => 'intermediate',
            'tags' => ['JavaScript', 'ES6', 'Programming', 'Web Development'],
            'is_free' => false,
            'is_published' => false,
            'order_index' => 3,
        ]);

        $this->command->info('Sample courses, lessons, topics, and code examples created successfully!');
    }
}
