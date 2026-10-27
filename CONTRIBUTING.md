# Contributing to lmlayer

Thanks for taking the time to improve `lmlayer`.

## Before you open a pull request

1. Keep the change focused: one fix or feature per pull request.
2. Run the smoke check:

   ```bash
   python tests/smoke_import.py
   ```

3. If you changed behavior, configuration, or the HTTP surface, update the relevant docs (`README.md`, `.env.example`, `docs/project-status.md`).
4. Do not commit `.env`, local databases, model files, or build artifacts.

## Reporting issues

Include what you ran, the configuration that matters (`MODEL`, `DB`, `TOOL`), the upstream server and version, and the complete error output. Remove API keys and other secrets first.

## Pull request scope

- Bug fixes and documentation corrections are always welcome.
- For a new feature, open an issue first so the behavior and scope can be agreed on.
- Dependency changes should explain why the existing dependency cannot do the job.

## License

By contributing, you agree that your contribution is licensed under the project's [AGPL-3.0](LICENSE) license.
