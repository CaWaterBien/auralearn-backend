<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Models\RagDocument;

class ForceCreateTestData extends Command
{
    protected $signature = 'rag:force-test-data';
    protected $description = 'Force create test RAG data for immediate testing';

    public function handle(): int
    {
        $this->info('🔥 FORCE CREATING TEST DATA...');

        // Clear existing data
        RagDocument::truncate();

        $testDocuments = [
            [
                'source' => 'html_basics.txt',
                'document_type' => 'html',
                'content' => 'HTML is the standard markup language for creating web pages. It uses elements like headings, paragraphs, links, and images.',
                'chunk_text' => 'HTML is the standard markup language for creating web pages. It uses elements like headings, paragraphs, links, and images.',
                'chunk_index' => 0,
                'metadata' => ['topic' => 'html_fundamentals'],
                'embedding' => array_fill(0, 1024, 0.1), // Mock embedding
                'embedding_dimensions' => 1024,
                'embedding_model' => 'BAAI/bge-multilingual-gemma2'
            ],
            [
                'source' => 'css_basics.txt', 
                'document_type' => 'css',
                'content' => 'CSS is used to style HTML elements. You can change colors, fonts, layouts, and more using CSS properties and selectors.',
                'chunk_text' => 'CSS is used to style HTML elements. You can change colors, fonts, layouts, and more using CSS properties and selectors.',
                'chunk_index' => 0,
                'metadata' => ['topic' => 'css_fundamentals'],
                'embedding' => array_fill(0, 1024, 0.2), // Different mock embedding
                'embedding_dimensions' => 1024,
                'embedding_model' => 'BAAI/bge-multilingual-gemma2'
            ],
            [
                'source' => 'html_headings.txt',
                'document_type' => 'html', 
                'content' => 'HTML headings are created using h1, h2, h3, h4, h5, and h6 elements. h1 is the most important heading and h6 is the least important.',
                'chunk_text' => 'HTML headings are created using h1, h2, h3, h4, h5, and h6 elements. h1 is the most important heading and h6 is the least important.',
                'chunk_index' => 0,
                'metadata' => ['topic' => 'html_headings'],
                'embedding' => array_fill(0, 1024, 0.3), // Another mock embedding
                'embedding_dimensions' => 1024,
                'embedding_model' => 'BAAI/bge-multilingual-gemma2'
            ]
        ];

        foreach ($testDocuments as $doc) {
            RagDocument::create($doc);
            $this->info("✅ Created: {$doc['source']}");
        }

        $this->info("🎉 FORCE DATA CREATION COMPLETE!");
        $this->info("Created " . count($testDocuments) . " test documents");
        
        return 0;
    }
}
