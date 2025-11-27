<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('code_examples', function (Blueprint $table) {
            $table->id();
            $table->foreignId('lesson_id')->nullable()->constrained()->onDelete('cascade');
            $table->foreignId('topic_id')->nullable()->constrained()->onDelete('cascade');
            $table->string('title');
            $table->text('description')->nullable();
            $table->string('language')->default('html'); // html, css, javascript, php, etc.
            $table->longText('initial_code'); // Default code shown in editor
            $table->longText('solution_code')->nullable(); // Solution/expected code
            $table->text('hints')->nullable(); // Hints for the user
            $table->integer('is_interactive')->default(1);
            $table->integer('show_preview')->default(1);
            $table->integer('allow_reset')->default(1);
            $table->json('test_cases')->nullable(); // For validating user code
            $table->integer('order_index')->default(0);
            $table->timestamps();
            
            $table->index('lesson_id');
            $table->index('topic_id');
            $table->index('order_index');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('code_examples');
    }
};
