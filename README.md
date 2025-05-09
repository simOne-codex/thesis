# Python ML template
A simple template to bootstrap Python machine learning projects, maintaining a standard structure.
It should be slightly faster than starting from scratch each time.

## Getting Started

To use this template, follow these steps:

1. Click the "Use this template" button at the top of the repository and follow the procedure.

2. Clone your new repository to your local machine:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   ```

3. Navigate to the project directory. Optional but recommended: create and activate a Python virtual environment to isolate your project's dependencies.
E.g.:

   ```bash
   cd your-repo
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
   ```
5. The template provides a simple "self-destructing" initialization script, `init.py`, that automatically provides the necessary information to generate a fully functional python package (project name, author, ...).
From a python environment, or any other means, this script can be launched as easily as:

    ```bash
    # launch and follow the prompts
    python init.py
    ```

6. Install the required dependencies:

   ```bash
    # Install the bare minimum, editable is usually preferred when developing
   pip install -e .
   # Install extras
   pip install -e .[dev,docs,test]
   ```


7. You're good to go! Of course, you can further customize it to your liking.

> **Note**
>
> The `init.py` script is self-contained and will delete itself once the procedure is completed. It is absolutely safe to delete if you prefer to edit the files manually.

## Extra goodies

If you are using VS Code as your editor of choice, you can use the following
snippet in your `settings.json` file to format and sort imports on save.

```json
{
    "[python]": {
        "diffEditor.ignoreTrimWhitespace": false,
        "editor.wordBasedSuggestions": "off",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        },
        "editor.defaultFormatter": "charliermarsh.ruff",
    },
}
```
Of course, this is completely optional.
