<?php

namespace Tests\Feature;

use App\Models\Admin;
use Illuminate\Support\Facades\Hash;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class AdminAuthTest extends TestCase
{
    use RefreshDatabase;

    public function test_admin_can_login_and_fetch_me_and_logout(): void
    {
        $admin = Admin::create([
            'name' => 'Test Admin',
            'email' => 'admin@example.com',
            'password' => Hash::make('secret123'),
        ]);

        $res = $this->postJson('/api/admin/login', [
            'email' => 'admin@example.com',
            'password' => 'secret123',
        ]);
        $res->assertOk();

        $me = $this->getJson('/api/admin/me');
        $me->assertOk()->assertJsonPath('admin.email', 'admin@example.com');

        $logout = $this->postJson('/api/admin/logout');
        $logout->assertOk();

        $me2 = $this->getJson('/api/admin/me');
        $me2->assertStatus(401);
    }
} 