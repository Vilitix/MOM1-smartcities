module.exports = {
    apps: [{
        name: "mom1-smartcities",
        script: "app.py",
        interpreter: "python3",
        // Safe memory limit for phone-hosting in Lite Mode
        max_memory_restart: '400M',
        // High timeout to ensure slow phone restarts don't trigger "crash" loops incorrectly
        kill_timeout: 15000,
        // Restart delay: avoid draining battery by constantly rebooting if it fails
        restart_delay: 8000,
        // Exponential backoff for safety
        exp_backoff_restart_delay: 2000,
        // Fork mode is most stable for Python on Android/Termux
        exec_mode: 'fork',
        instances: 1,
        env: {
            LITE_MODE: "true",      // Explicitly set to Lite Mode as requested
            DEPLOYMENT: "true",   // Triggers Waitress instead of Flask Debugger
            PYTHONUNBUFFERED: "1"
        }
    }]
};
