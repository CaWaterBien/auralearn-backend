<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Services\RagEmbeddingService;
use App\Models\RagDocument;
use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\Log;

class IngestRagDocuments extends Command
{
    protected $signature = 'rag:ingest
                            {source : Source path (file or directory)}
                            {--type=text : Document type (text, html, css, lesson, activity)}
                            {--clear : Clear existing documents before ingesting}
                            {--course-id= : Associate with specific course ID}
                            {--lesson-id= : Associate with specific lesson ID}';

    protected $description = 'Ingest documents into the RAG system for AuraBot';

    private RagEmbeddingService $embeddingService;

    public function __construct(RagEmbeddingService $embeddingService)
    {
        parent::__construct();
        $this->embeddingService = $embeddingService;
    }

    public function handle(): int
    {
        $source = $this->argument('source');
        $documentType = $this->option('type');
        $shouldClear = $this->option('clear');
        $courseId = $this->option('course-id');
        $lessonId = $this->option('lesson-id');

        // Clear existing documents if requested
        if ($shouldClear) {
            $this->info('Clearing existing RAG documents...');
            $deletedCount = RagDocument::query()->delete();
            $this->info("Deleted {$deletedCount} existing documents.");
        }

        // Build metadata
        $metadata = array_filter([
            'course_id' => $courseId,
            'lesson_id' => $lessonId,
            'ingested_at' => now()->toISOString(),
            'ingested_by' => 'artisan_command'
        ]);

        if (is_dir($source)) {
            return $this->ingestDirectory($source, $documentType, $metadata);
        } elseif (is_file($source)) {
            return $this->ingestFile($source, $documentType, $metadata);
        } else {
            $this->error("Source path does not exist: {$source}");
            return 1;
        }
    }

    private function ingestDirectory(string $directory, string $documentType, array $metadata): int
    {
        $this->info("Ingesting directory: {$directory}");
        
        $files = File::allFiles($directory);
        $totalFiles = count($files);
        $processedFiles = 0;
        $totalChunks = 0;

        $progressBar = $this->output->createProgressBar($totalFiles);
        $progressBar->start();

        foreach ($files as $file) {
            try {
                $filePath = $file->getRealPath();
                $relativePath = str_replace($directory, '', $filePath);
                
                // Skip non-text files
                if (!$this->isTextFile($filePath)) {
                    $progressBar->advance();
                    continue;
                }

                $chunks = $this->ingestFile($filePath, $documentType, array_merge($metadata, [
                    'relative_path' => $relativePath,
                    'file_size' => filesize($filePath)
                ]));

                $totalChunks += $chunks;
                $processedFiles++;

            } catch (\Exception $e) {
                $this->error("Failed to process file {$file->getRealPath()}: " . $e->getMessage());
                Log::error('File ingestion error', [
                    'file' => $file->getRealPath(),
                    'error' => $e->getMessage()
                ]);
            }

            $progressBar->advance();
        }

        $progressBar->finish();
        $this->newLine();
        $this->info("Processed {$processedFiles} files, created {$totalChunks} document chunks.");

        return 0;
    }

    private function ingestFile(string $filePath, string $documentType, array $metadata): int
    {
        if (!$this->isTextFile($filePath)) {
            return 0;
        }

        $content = File::get($filePath);
        $fileName = basename($filePath);

        // Auto-detect document type based on file extension
        $extension = strtolower(pathinfo($filePath, PATHINFO_EXTENSION));
        if (in_array($extension, ['html', 'htm'])) {
            $documentType = 'html';
        } elseif (in_array($extension, ['css'])) {
            $documentType = 'css';
        } elseif (in_array($extension, ['js', 'javascript'])) {
            $documentType = 'javascript';
        } elseif (in_array($extension, ['md', 'markdown'])) {
            $documentType = 'markdown';
        }

        try {
            $chunksCreated = $this->embeddingService->ingestDocument(
                $content,
                $fileName,
                $documentType,
                array_merge($metadata, [
                    'file_extension' => $extension,
                    'file_size' => strlen($content),
                    'ingestion_method' => 'artisan_command'
                ])
            );

            $this->info("Ingested {$fileName}: {$chunksCreated} chunks created");
            return $chunksCreated;

        } catch (\Exception $e) {
            $this->error("Failed to ingest {$fileName}: " . $e->getMessage());
            return 0;
        }
    }

    private function isTextFile(string $filePath): bool
    {
        $allowedExtensions = [
            'txt', 'md', 'markdown', 'html', 'htm', 'css', 'js', 'json',
            'xml', 'yml', 'yaml', 'php', 'py', 'java', 'cpp', 'c', 'h'
        ];

        $extension = strtolower(pathinfo($filePath, PATHINFO_EXTENSION));
        return in_array($extension, $allowedExtensions);
    }
}

