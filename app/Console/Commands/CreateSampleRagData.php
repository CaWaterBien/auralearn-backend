<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Services\RagEmbeddingService;

class CreateSampleRagData extends Command
{
    protected $signature = 'rag:create-samples';
    protected $description = 'Create sample RAG documents for HTML/CSS learning';

    private RagEmbeddingService $embeddingService;

    public function __construct(RagEmbeddingService $embeddingService)
    {
        parent::__construct();
        $this->embeddingService = $embeddingService;
    }

    public function handle(): int
    {
        $this->info('Creating sample RAG documents for AuraBot...');

        $sampleDocuments = [
            [
                'content' => "HTML Basics: HTML (HyperText Markup Language) is the standard markup language for creating web pages. Every HTML document starts with a DOCTYPE declaration, followed by the html element which contains the head and body sections. The head contains metadata about the document, while the body contains the visible content. Basic HTML elements include headings (h1-h6), paragraphs (p), links (a), images (img), and lists (ul, ol, li). Always remember to close your tags properly and use semantic HTML elements when possible.",
                'source' => 'html_basics_guide.txt',
                'type' => 'html',
                'metadata' => ['topic' => 'html_fundamentals', 'difficulty' => 'beginner']
            ],
            [
                'content' => "CSS Fundamentals: CSS (Cascading Style Sheets) is used to style HTML elements. CSS works with selectors that target HTML elements, and properties that define how those elements should look. The basic syntax is selector { property: value; }. Common selectors include element selectors (p, h1), class selectors (.classname), and ID selectors (#idname). Important CSS properties include color, font-size, margin, padding, width, height, display, and position. The box model is fundamental to understanding CSS layout.",
                'source' => 'css_fundamentals_guide.txt',
                'type' => 'css',
                'metadata' => ['topic' => 'css_fundamentals', 'difficulty' => 'beginner']
            ],
            [
                'content' => "HTML Forms: Forms are essential for user interaction on websites. The form element wraps all form controls. Common form elements include input (with types like text, email, password, submit), textarea for multi-line text, select for dropdown menus, and button elements. Always include labels for accessibility using the label element or aria-label attribute. Form validation can be done with HTML5 attributes like required, pattern, min, max, and type-specific validation.",
                'source' => 'html_forms_guide.txt',
                'type' => 'html',
                'metadata' => ['topic' => 'forms', 'difficulty' => 'intermediate']
            ],
            [
                'content' => "CSS Flexbox: Flexbox is a powerful layout method for creating flexible and responsive layouts. Use display: flex on a container to make it a flex container. Key properties include justify-content (horizontal alignment), align-items (vertical alignment), flex-direction (row or column), and flex-wrap (wrapping behavior). Flex items can use flex-grow, flex-shrink, and flex-basis to control their sizing behavior. Flexbox is perfect for centering content, creating equal-height columns, and responsive navigation bars.",
                'source' => 'css_flexbox_guide.txt',
                'type' => 'css',
                'metadata' => ['topic' => 'flexbox', 'difficulty' => 'intermediate']
            ],
            [
                'content' => "CSS Grid: CSS Grid is a two-dimensional layout system that allows you to create complex layouts with rows and columns. Use display: grid to create a grid container. Define the grid structure with grid-template-columns and grid-template-rows. Position items using grid-column and grid-row properties. Grid areas can be named using grid-template-areas for easier positioning. CSS Grid is excellent for page layouts, card grids, and complex responsive designs.",
                'source' => 'css_grid_guide.txt',
                'type' => 'css',
                'metadata' => ['topic' => 'grid', 'difficulty' => 'intermediate']
            ],
            [
                'content' => "Responsive Design: Responsive design ensures websites work well on all device sizes. Use mobile-first approach by starting with mobile styles and using min-width media queries. Key techniques include flexible units (%, em, rem, vw, vh), responsive images with max-width: 100%, and flexible layouts with Flexbox or Grid. Common breakpoints are 768px for tablets and 1024px for desktop. Use viewport meta tag: <meta name='viewport' content='width=device-width, initial-scale=1'>.",
                'source' => 'responsive_design_guide.txt',
                'type' => 'css',
                'metadata' => ['topic' => 'responsive_design', 'difficulty' => 'intermediate']
            ],
            [
                'content' => "HTML Semantic Elements: Semantic HTML elements provide meaning to the content structure. Use header for page/section headers, nav for navigation menus, main for main content, article for standalone content, section for thematic groupings, aside for sidebar content, and footer for page/section footers. These elements improve accessibility, SEO, and code readability. Screen readers and search engines rely on semantic markup to understand content structure.",
                'source' => 'html_semantic_guide.txt',
                'type' => 'html',
                'metadata' => ['topic' => 'semantic_html', 'difficulty' => 'intermediate']
            ],
            [
                'content' => "Common HTML/CSS Debugging: When debugging HTML/CSS issues, check for common problems: unclosed tags, typos in class/ID names, missing CSS selectors, specificity conflicts, and browser compatibility issues. Use browser developer tools to inspect elements, view computed styles, and test changes live. Validate your HTML and CSS using online validators. Common CSS issues include forgetting to clear floats, incorrect box-sizing, and z-index stacking context problems.",
                'source' => 'debugging_guide.txt',
                'type' => 'tutorial',
                'metadata' => ['topic' => 'debugging', 'difficulty' => 'beginner']
            ],
            [
                'content' => "CSS Animations and Transitions: CSS animations bring life to web pages. Transitions create smooth changes between states using the transition property. Animations use @keyframes to define animation sequences and the animation property to apply them. Key animation properties include animation-duration, animation-timing-function, animation-delay, animation-iteration-count, and animation-direction. Use transform for performant animations (translate, scale, rotate) and avoid animating layout properties like width and height.",
                'source' => 'css_animations_guide.txt',
                'type' => 'css',
                'metadata' => ['topic' => 'animations', 'difficulty' => 'advanced']
            ],
            [
                'content' => "HTML Best Practices: Write clean, maintainable HTML by using proper indentation, meaningful element names, and consistent formatting. Always include alt attributes for images, use heading hierarchy properly (h1 -> h2 -> h3), and provide form labels for accessibility. Validate your HTML regularly, minimize inline styles, and use semantic elements appropriately. Keep your HTML structure logical and easy to understand for both humans and machines.",
                'source' => 'html_best_practices.txt',
                'type' => 'tutorial',
                'metadata' => ['topic' => 'best_practices', 'difficulty' => 'intermediate']
            ]
        ];

        $progressBar = $this->output->createProgressBar(count($sampleDocuments));
        $progressBar->start();

        $totalChunks = 0;

        foreach ($sampleDocuments as $doc) {
            try {
                $chunks = $this->embeddingService->ingestDocument(
                    $doc['content'],
                    $doc['source'],
                    $doc['type'],
                    $doc['metadata']
                );

                $totalChunks += $chunks;
                $this->info("\nIngested {$doc['source']}: {$chunks} chunks");

            } catch (\Exception $e) {
                $this->error("\nFailed to ingest {$doc['source']}: " . $e->getMessage());
            }

            $progressBar->advance();
        }

        $progressBar->finish();
        $this->newLine();
        $this->info("Sample data creation complete! Total chunks created: {$totalChunks}");

        return 0;
    }
}

