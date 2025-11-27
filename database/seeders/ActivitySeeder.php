<?php

namespace Database\Seeders;

use Illuminate\Database\Seeder;
use App\Models\Activity;
use App\Models\Lesson;

class ActivitySeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        // Get lessons to attach activities to
        $htmlIntroLesson = Lesson::where('slug', 'html-introduction')->first();
        
        if ($htmlIntroLesson) {
            // Activity 1: HTML Hello World - Complete beginner activity
            Activity::create([
                'lesson_id' => $htmlIntroLesson->id,
                'title' => 'HTML Hello World',
                'description' => 'Create your first HTML page',
                'activity_type' => 'coding',
                'instructions' => '1. Start with HTML5 DOCTYPE\n2. Create basic structure\n3. Add title and heading\n4. Add paragraph',
                'metadata' => [
                        'initial_code' => '<!DOCTYPE html>
<html>
<head>
    <!-- Add title here -->
</head>
<body>
    <!-- Add heading and paragraph here -->
</body>
</html>',
                    'expected_output' => 'Hello, World! This is my first HTML page.',
                    'validation_criteria' => [
                        'required_elements' => ['html', 'head', 'body', 'title', 'h1', 'p'],
                        'structure_checks' => [
                            ['type' => 'doctype'],
                            ['type' => 'nested', 'parent' => 'head', 'child' => 'title'],
                            ['type' => 'nested', 'parent' => 'body', 'child' => 'h1'],
                            ['type' => 'nested', 'parent' => 'body', 'child' => 'p']
                        ]
                    ],
                    'hints' => [
                        'Add a <title> tag inside the <head> section',
                        'Use <h1> for the main heading "Hello, World!"',
                        'Add a <p> tag with text about your first HTML page',
                        'Make sure all tags are properly closed'
                    ],
                    'solution_example' => '<!DOCTYPE html>
<html>
<head>
    <title>My First Page</title>
</head>
<body>
    <h1>Hello, World!</h1>
    <p>This is my first HTML page.</p>
</body>
</html>'
                ],
                'time_limit' => 15,
                'max_attempts' => 5,
                'passing_score' => 100,
                'points' => 100,
                'is_required' => true,
                'is_published' => true,
                'order_index' => 1
            ]);

            // Activity 2: HTML Document Structure
            Activity::create([
                'lesson_id' => $htmlIntroLesson->id,
                'title' => 'HTML Document Structure',
                'description' => 'Build a proper HTML document with multiple sections',
                'activity_type' => 'coding',
                'instructions' => '1. Create HTML5 document\n2. Add proper head section with title and meta tags\n3. Create header with site title\n4. Add main content area\n5. Include footer',
                'metadata' => [
                    'initial_code' => '<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <!-- Add title and viewport meta tag -->
</head>
<body>
    <!-- Add header, main, and footer sections -->
</body>
</html>',
                    'expected_output' => 'Welcome to My Website About Me This is the main content area. Contact us at info@example.com',
                    'validation_criteria' => [
                        'required_elements' => ['html', 'head', 'body', 'title', 'meta', 'header', 'main', 'footer'],
                        'required_attributes' => [
                            'html' => ['lang' => 'en'],
                            'meta' => ['charset' => 'UTF-8']
                        ],
                        'structure_checks' => [
                            ['type' => 'doctype'],
                            ['type' => 'order', 'first' => 'header', 'second' => 'main'],
                            ['type' => 'order', 'first' => 'main', 'second' => 'footer']
                        ]
                    ],
                    'hints' => [
                        'Add viewport meta tag: <meta name="viewport" content="width=device-width, initial-scale=1.0">',
                        'Use semantic HTML5 elements: <header>, <main>, <footer>',
                        'Put the header before main, and main before footer',
                        'Include meaningful content in each section'
                    ],
                    'solution_example' => '<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Website</title>
</head>
<body>
    <header>
        <h1>Welcome to My Website</h1>
    </header>
    <main>
        <h2>About Me</h2>
        <p>This is the main content area.</p>
    </main>
    <footer>
        <p>Contact us at info@example.com</p>
    </footer>
</body>
</html>'
                ],
                'time_limit' => 20,
                'max_attempts' => 5,
                'passing_score' => 100,
                'points' => 150,
                'is_required' => true,
                'is_published' => true,
                'order_index' => 2
            ]);

            // Activity 3: Text Formatting and Lists
            Activity::create([
                'lesson_id' => $htmlIntroLesson->id,
                'title' => 'Text Formatting and Lists',
                'description' => 'Practice using various HTML text formatting elements and lists',
                'activity_type' => 'coding',
                'instructions' => '1. Create a page about your favorite hobby\n2. Use headings (h1, h2, h3)\n3. Add formatted text (bold, italic, underline)\n4. Create both ordered and unordered lists\n5. Include a blockquote',
                'metadata' => [
                    'initial_code' => '<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Hobby</title>
</head>
<body>
    <!-- Add your content here -->
</body>
</html>',
                    'expected_output' => 'My Favorite Hobby Photography What I Love About Photography Equipment I Use Tips for Beginners As Ansel Adams once said The single most important component of a camera is the twelve inches behind it',
                    'validation_criteria' => [
                        'required_elements' => ['h1', 'h2', 'h3', 'strong', 'em', 'ul', 'ol', 'li', 'blockquote'],
                        'structure_checks' => [
                            ['type' => 'nested', 'parent' => 'ul', 'child' => 'li'],
                            ['type' => 'nested', 'parent' => 'ol', 'child' => 'li']
                        ]
                    ],
                    'hints' => [
                        'Use <h1> for the main title, <h2> for sections, <h3> for subsections',
                        'Use <strong> for important text, <em> for emphasis',
                        'Create <ul> for unordered lists, <ol> for ordered lists',
                        'Don\'t forget <li> elements inside your lists',
                        'Use <blockquote> for quotes'
                    ],
                    'solution_example' => '<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Hobby</title>
</head>
<body>
    <h1>My Favorite Hobby</h1>
    <h2>Photography</h2>
    
    <h3>What I Love About Photography</h3>
    <p>Photography allows me to <strong>capture moments</strong> and express my <em>creativity</em>.</p>
    
    <h3>Equipment I Use</h3>
    <ul>
        <li>DSLR Camera</li>
        <li>Prime Lens</li>
        <li>Tripod</li>
    </ul>
    
    <h3>Tips for Beginners</h3>
    <ol>
        <li>Learn the basics of composition</li>
        <li>Practice regularly</li>
        <li>Study light and shadows</li>
    </ol>
    
    <blockquote>
        <p>As Ansel Adams once said: "The single most important component of a camera is the twelve inches behind it."</p>
    </blockquote>
</body>
</html>'
                ],
                'time_limit' => 25,
                'max_attempts' => 5,
                'passing_score' => 100,
                'points' => 200,
                'is_required' => true,
                'is_published' => true,
                'order_index' => 3
            ]);
        }

        // Create activities for other lessons if they exist
        $htmlEditorsLesson = Lesson::where('slug', 'html-editors')->first();
        
        if ($htmlEditorsLesson) {
            // Activity: Links and Navigation
            Activity::create([
                'lesson_id' => $htmlEditorsLesson->id,
                'title' => 'Links and Navigation',
                'description' => 'Create a navigation menu with various types of links',
                'activity_type' => 'coding',
                'instructions' => '1. Create a navigation menu\n2. Add internal links to sections\n3. Include external link\n4. Add email link\n5. Create link to download file',
                'metadata' => [
                    'initial_code' => '<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Navigation Example</title>
</head>
<body>
    <!-- Create navigation and sections here -->
</body>
</html>',
                    'expected_output' => 'Navigation Home About Contact Services Web Design SEO Marketing Welcome to Our Website About Us Contact Us Visit Google Email us Download our brochure',
                    'validation_criteria' => [
                        'required_elements' => ['nav', 'ul', 'li', 'a', 'section'],
                        'required_attributes' => [
                            'a' => ['href' => '#about'],
                        ],
                        'structure_checks' => [
                            ['type' => 'nested', 'parent' => 'nav', 'child' => 'ul'],
                            ['type' => 'nested', 'parent' => 'ul', 'child' => 'li'],
                            ['type' => 'nested', 'parent' => 'li', 'child' => 'a']
                        ]
                    ],
                    'hints' => [
                        'Use <nav> element for navigation menu',
                        'Create internal links with href="#sectionid"',
                        'External links need full URLs like "https://google.com"',
                        'Email links use "mailto:email@example.com"',
                        'Use id attributes for sections you want to link to'
                    ],
                    'solution_example' => '<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Navigation Example</title>
</head>
<body>
    <nav>
        <ul>
            <li><a href="#home">Home</a></li>
            <li><a href="#about">About</a></li>
            <li><a href="#contact">Contact</a></li>
            <li><a href="#services">Services</a></li>
        </ul>
    </nav>
    
    <section id="home">
        <h1>Welcome to Our Website</h1>
    </section>
    
    <section id="about">
        <h2>About Us</h2>
    </section>
    
    <section id="services">
        <h2>Services</h2>
        <ul>
            <li>Web Design</li>
            <li>SEO</li>
            <li>Marketing</li>
        </ul>
    </section>
    
    <section id="contact">
        <h2>Contact Us</h2>
        <p><a href="https://google.com" target="_blank">Visit Google</a></p>
        <p><a href="mailto:info@example.com">Email us</a></p>
        <p><a href="brochure.pdf" download>Download our brochure</a></p>
    </section>
</body>
</html>'
                ],
                'time_limit' => 30,
                'max_attempts' => 5,
                'passing_score' => 100,
                'points' => 250,
                'is_required' => true,
                'is_published' => true,
                'order_index' => 1
            ]);
        }

        $this->command->info('Realistic HTML5 activities created successfully!');
    }
}
