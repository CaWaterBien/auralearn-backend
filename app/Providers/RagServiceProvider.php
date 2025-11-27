<?php

namespace App\Providers;

use Illuminate\Support\ServiceProvider;
use App\Services\NebiusClient;
use App\Services\RagEmbeddingService;
use App\Services\AuraBotRagService;
use App\Services\AiValidationService;

class RagServiceProvider extends ServiceProvider
{
    /**
     * Register services.
     */
    public function register(): void
    {
        // Register Nebius Client as singleton
        $this->app->singleton(NebiusClient::class, function ($app) {
            return new NebiusClient();
        });

        // Register RAG Embedding Service
        $this->app->singleton(RagEmbeddingService::class, function ($app) {
            return new RagEmbeddingService($app->make(NebiusClient::class));
        });

        // Register AuraBot RAG Service
        $this->app->singleton(AuraBotRagService::class, function ($app) {
            return new AuraBotRagService(
                $app->make(NebiusClient::class),
                $app->make(RagEmbeddingService::class)
            );
        });

        // Register AI Validation Service
        $this->app->singleton(AiValidationService::class, function ($app) {
            return new AiValidationService($app->make(NebiusClient::class));
        });
    }

    /**
     * Bootstrap services.
     */
    public function boot(): void
    {
        // Register console commands
        if ($this->app->runningInConsole()) {
            $this->commands([
                \App\Console\Commands\IngestRagDocuments::class,
                \App\Console\Commands\IngestCourseContent::class,
                \App\Console\Commands\CreateSampleRagData::class,
                \App\Console\Commands\TestRagSystem::class,
                \App\Console\Commands\TestSystemWithoutAPIs::class,
                \App\Console\Commands\ForceCreateTestData::class,
            ]);
        }
    }
}
