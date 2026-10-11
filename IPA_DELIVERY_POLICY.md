# IPA delivery safeguard

For RIFE60GhostGuard builds, the unsigned IPA must be delivered through a temporary GitHub Release only.

- Do not upload or attach the IPA to ChatGPT Library or the conversation.
- Do not publish the IPA as a GitHub Actions workflow artifact.
- Create a uniquely tagged GitHub Release containing the IPA after validation.
- Provide the release URL to the user.
- Delete the temporary release and its IPA asset 15 minutes after publication.
- Diagnostics and build logs may remain Actions artifacts; they must not contain the IPA itself.

The release publication and 15-minute deletion steps live in `.github/workflows/build-unsigned-ipa.yml`.
