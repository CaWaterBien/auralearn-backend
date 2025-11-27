<?php

require_once 'vendor/autoload.php';

$app = require_once 'bootstrap/app.php';
$app->boot();

use Illuminate\Support\Facades\DB;

echo "🔥 FORCE INSERTING TEST DATA...\n";

// Clear existing data
DB::table('rag_documents')->truncate();
echo "✅ Cleared existing data\n";

// Insert test data directly
$inserted = DB::table('rag_documents')->insert([
    [
        'source' => 'html_basics.txt',
        'document_type' => 'html',
        'content' => 'HTML is the standard markup language for creating web pages. It uses elements like headings, paragraphs, links, and images.',
        'chunk_text' => 'HTML is the standard markup language for creating web pages. It uses elements like headings, paragraphs, links, and images.',
        'chunk_index' => 0,
        'metadata' => json_encode(['topic' => 'html_fundamentals']),
        'embedding' => json_encode(array_fill(0, 1024, 0.1)),
        'embedding_dimensions' => 1024,
        'embedding_model' => 'BAAI/bge-multilingual-gemma2',
        'created_at' => now(),
        'updated_at' => now()
    ],
    [
        'source' => 'css_basics.txt',
        'document_type' => 'css', 
        'content' => 'CSS is used to style HTML elements. You can change colors, fonts, layouts, and more using CSS properties and selectors.',
        'chunk_text' => 'CSS is used to style HTML elements. You can change colors, fonts, layouts, and more using CSS properties and selectors.',
        'chunk_index' => 0,
        'metadata' => json_encode(['topic' => 'css_fundamentals']),
        'embedding' => json_encode(array_fill(0, 1024, 0.2)),
        'embedding_dimensions' => 1024,
        'embedding_model' => 'BAAI/bge-multilingual-gemma2',
        'created_at' => now(),
        'updated_at' => now()
    ],
    [
        'source' => 'html_headings.txt',
        'document_type' => 'html',
        'content' => 'HTML headings use h1, h2, h3, h4, h5, h6 elements. h1 is most important for main titles, h2 for section headers, etc.',
        'chunk_text' => 'HTML headings use h1, h2, h3, h4, h5, h6 elements. h1 is most important for main titles, h2 for section headers, etc.',
        'chunk_index' => 0,
        'metadata' => json_encode(['topic' => 'html_headings']),
        'embedding' => json_encode(array_fill(0, 1024, 0.4)),
        'embedding_dimensions' => 1024,
        'embedding_model' => 'BAAI/bge-multilingual-gemma2',
        'created_at' => now(),
        'updated_at' => now()
    ]
]);

echo "✅ Inserted test documents\n";
echo "📊 Total documents in database: " . DB::table('rag_documents')->count() . "\n";
echo "🎉 TEST DATA READY FOR AURABOT!\n";
