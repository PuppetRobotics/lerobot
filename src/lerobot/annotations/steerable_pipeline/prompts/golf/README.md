# Golf subtask prompts

Prompt variants for annotating golf (putting) datasets with `lerobot-annotate`.
They replace the two prompts that produce subtasks:

| File                        | Replaces                       | What changes                                                                                                            |
| --------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `plan_subtask_describe.txt` | `../plan_subtask_describe.txt` | Narrates club and ball motion (where the club head moves, when the ball starts rolling) instead of grasps and releases. |
| `plan_subtasks.txt`         | `../plan_subtasks.txt`         | Segments into a fixed list of six labels, with boundaries defined by the club and the ball.                             |

Both files also add a SCENE block that names the objects: the club (not a
hammer, mallet, racket and so on), the ball and the mat. Without it, the
default prompts labelled the club as a hammer, racket or scraper in about a
third of episodes.

## The six labels

| Label                 | Meaning                                                 |
| --------------------- | ------------------------------------------------------- |
| `address ball`        | club head moves to rest just behind the ball            |
| `swing club back`     | backswing: club head moves away from the ball           |
| `strike ball`         | forward swing through impact, until the ball is rolling |
| `hold follow-through` | club continues past impact, slows and is held           |
| `reposition club`     | club moves without interacting with the ball            |
| `push ball with club` | club shoves the ball without a backswing or impact      |

The prompt tells the model to use only these strings, but nothing in the
pipeline checks this. Check the output, or map unknown labels to
`reposition club` afterwards.

## How to use them

Pass `--plan.prompts_dir=golf` to `lerobot-annotate`. The two files in this
folder replace the defaults of the same name; every other prompt (memory,
task paraphrases, relabel, VQA, interjections) keeps its default. The log
line `plan: prompts_dir=... replaces plan_subtask_describe, plan_subtasks`
confirms the variant is active.

These are the settings used for
`puppet-robotics/golf_3_servo_annotated_custom_prompts`:

```bash
uv run lerobot-annotate \
  --repo_id <source_repo_id> \
  --new_repo_id <target_repo_id> \
  --plan.prompts_dir=golf \
  --plan.derive_task_from_video=off \
  --plan.frames_per_second=4 \
  --plan.max_frames_per_prompt=140 \
  --plan.min_subtask_seconds=0.5 \
  --plan.plan_max_steps=8 \
  --vlm.client_concurrency=4 \
  --executor.episode_parallelism=4 \
  --push_to_hub=true
```

Add your VLM server flags (`--vlm.serve_command`, `--vlm.api_base`,
`--vlm.chat_template_kwargs` and so on) as usual.

`--plan.prompts_dir` also takes a path, so you can copy this folder, edit the
copy for another setup and pass `--plan.prompts_dir=/path/to/my_prompts`. A
variant folder only needs the files it changes. Names must match a default
template (`plan_subtasks.txt`, `plan_subtask_describe.txt`, ...); a misspelled
name or a missing folder stops the run at startup.

A `LEROBOT_PROMPT_OVERRIDE_<name>` environment variable still takes
precedence over both the variant and the default.

### Installs without `--plan.prompts_dir`

Older lerobot versions (for example the 0.7.0 release) only read files
directly in `prompts/`. There, copy the golf prompts over the defaults for the
run and restore them afterwards:

```bash
P=$(uv run python -c "import lerobot.annotations.steerable_pipeline.prompts as p, os; print(os.path.dirname(p.__file__))")
cp "$P/plan_subtasks.txt" "$P/plan_subtasks.txt.orig"
cp "$P/plan_subtask_describe.txt" "$P/plan_subtask_describe.txt.orig"
cp /path/to/golf/plan_subtasks.txt /path/to/golf/plan_subtask_describe.txt "$P/"
# ... run lerobot-annotate without --plan.prompts_dir ...
mv "$P/plan_subtasks.txt.orig" "$P/plan_subtasks.txt"
mv "$P/plan_subtask_describe.txt.orig" "$P/plan_subtask_describe.txt"
```

Copy (`cp`), don't edit the installed files in place: uv may hard-link them to
its package cache, so an in-place edit also changes every other environment
installed from that cache.

### Why these settings

- **`--plan.derive_task_from_video=off`**: tasks shorter than
  `derive_task_min_words` (3) are replaced by a task the VLM guesses from the
  video, with no hint. "Play golf" is two words, so without this flag the
  prompts never see it.
- **`--plan.frames_per_second=4` and `--plan.max_frames_per_prompt=140`**:
  a backswing or strike often lasts under a second, so 2 fps can miss the
  impact frame. Episodes longer than `max_frames_per_prompt / frames_per_second`
  seconds are split into windows that are labelled independently, which cuts
  a subtask at every window edge. 140 frames at 4 fps covers 35 s, which is
  longer than any episode in `golf_3_servo_expected`. Raise it for longer
  episodes and check that the VLM server accepts that many images per request
  (vLLM `--limit-mm-per-prompt`).
- **`--plan.min_subtask_seconds=0.5`**: the default 1.5 s merges the
  backswing into the strike.
- **Lower concurrency**: each request carries up to 7 contact sheets. With
  the defaults (16 episodes, 16 requests each) requests wait in the vLLM queue
  long enough to time out. Raise `--vlm.request_timeout_s` (default 600) if
  they still do.

## Known limitations

- **`min_subtask_seconds` and `plan_max_steps` are not enforced.** They only
  appear in the prompt text. The golf run had 20 segments shorter than 0.5 s
  and 2 episodes with more than 8 segments.
- **Labels vary between runs.** The VLM samples at `temperature=0.2`; the
  same episode got 9 segments in one run and 16 in another.
- **The seeded relabel pass is not adapted.** `plan_subtask_relabel.txt`
  does not know the six labels and may rewrite them into free text. Keep
  `--plan.subtask_seeded_relabel` off with these prompts, or adapt that
  prompt too.
- **Gripper-based boundary rules are still appended.** The plan module adds a
  fixed EVENT BOUNDARIES block (objects becoming held or released) to the
  describe and segment prompts in code, so `prompts_dir` cannot remove it.
  The golf prompts tell the model the gripper does not open or close, so the
  two texts disagree. The golf run used both; removing the block would need a
  code change.
- **The scene description is specific to the Puppet golf setup** (putter
  with a black head, white ball, green mat). Change the SCENE block in both
  files for a different club, ball or surface.
- **Episodes without a stroke.** In `golf_3_servo_annotated_custom_prompts`,
  75 of 172 episodes have no stroke, mostly a single `reposition club`
  segment. Several of those come in runs of consecutive episodes and should
  be checked against the video before trusting them.
