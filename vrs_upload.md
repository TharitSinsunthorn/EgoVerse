# VRS Metadata Collector and S3 Uploader

This script processes a directory of `.vrs` and `.vrs.json` files by:

* Launching `vrsplayer` for visual inspection.
* Prompting the user for metadata via a simple GUI.
* Renaming the files using a consistent naming scheme.
* Saving a per-recording metadata CSV.
* Uploading all renamed files and metadata to S3 **after all files are processed**.

---

## Usage

```bash
python upload_vrs_to_s3.py
```

You will be prompted to:

1. **Enter a task name** (e.g., `pour_milk`).
2. **Select a folder** containing `.vrs` and `.vrs.json` files.
3. **Optionally enter custom metadata keys** (comma-separated, e.g., `object,room`).
4. For each `.vrs` file:

   * `vrsplayer` will launch for you to inspect the recording.
   * A pop-up GUI form will appear for you to enter metadata including:

     * `collector`, `lab`, `scene`, `recording_number`
     * plus any custom fields you added
   * Click **Submit** to finish that recording.
5. Once all recordings are processed:

   * The files will be renamed.
   * A corresponding `_meta.csv` file will be saved for each.
   * All renamed files will be uploaded to:

```
s3://rldb/raw/{task_name}/
```

---

## Notes

* Ensure that your AWS credentials are configured before running the script.
* You can do this by running:

```bash
aws configure
```

* For detailed setup instructions, see [`egoverse.md`](./egoverse.md).
* Files without a matching `.vrs.json` are skipped.
* Uploading is deferred until **all recordings** have been processed.
* `vrsplayer` must be installed and available on your system path.
