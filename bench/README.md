# Benchmarks

How DualOCR compares with the two engines it combines, each used alone, on
five public datasets. Every system reads the same page image, and all of them
come from one engine pass per page, so the only difference between "Surya
alone" and "DualOCR" is DualOCR's alignment and voting.

In short: **on Hebrew print DualOCR is more accurate than either engine
alone. On English, Surya alone is just as good.** Neither engine reads Rashi
script or vowel-pointed (niqqud) text well, and DualOCR can't do much better
than its best engine there.

Run on 24-25 November 2026 on an M4 Pro Mac: DualOCR at commit `f637191`,
Surya 0.24.0 with its `surya-ocr-2.5` model on llama.cpp build 12359,
Tesseract 5.5 with the `tessdata_best` models.

## Results

| Dataset | Pages | Metric | Tesseract alone | Surya alone | DualOCR Fast | DualOCR Balanced (default) |
|---|---|---|---|---|---|---|
| Hebrew books, square script (Wikisource) | 34 | words right, in order | 66.0 | 73.8 | **76.2** | 75.6 |
| Hebrew books, Rashi script | 13 | words right, in order | 13.9 | 40.7 | **41.2** | 40.1 |
| Hebrew books with niqqud | 10 | words right, in order | 26.6 | **60.6** | 60.1 | 53.3 |
| Rendered Hebrew pages (Surya's rec_bench) | 10 | characters right | 98.5 | 98.8 | **99.0** | 98.8 |
| Rendered English pages (Surya's rec_bench) | 40 | Surya's line score | 89.7 | **97.5** | 97.0 | 96.9 |
| Noisy scanned forms (FUNSD) | 50 | word F1 | 49.7 | **80.0** | 79.3 | 79.0 |
| 1910s letters, typed and handwritten (olmOCR-bench `old_scans`) | 50 | tests passed | 15.9 | 31.7 | **32.5** | 32.1 |

"Tesseract alone" is plain `tesseract -l heb+eng` on the page as given.
All numbers are percentages. The full table, with more metrics per dataset, is
[`results/summary.md`](results/summary.md).

Page by page, DualOCR Fast against Surya alone (with a 95% bootstrap
interval for the average difference per page):

| Dataset | DualOCR better | Worse | Same | Average difference |
|---|---|---|---|---|
| Hebrew books | 38 pages | 8 | 11 | +1.6 points (+0.8 to +2.3) |
| Rendered Hebrew pages | 6 | 2 | 2 | +1.1 (−0.4 to +2.6) |
| Rendered English pages | 4 | 19 | 17 | −0.2 (−1.3 to +1.2) |

What the numbers say:

- **Hebrew:** DualOCR wins most pages against Surya alone, and Tesseract
  alone is far behind on real scans (DualOCR is ahead by 24 points per page on
  the book pages).
- **English:** Surya alone is about as good. On the rendered pages DualOCR
  loses small amounts on many pages, for example by dropping standalone "—"
  and "•" that Surya keeps. On forms it keeps text that only Tesseract saw, which
  finds a little more of the page but adds words the answer key doesn't have.
- **Rashi script and niqqud:** Tesseract's Hebrew model hardly reads Rashi
  script (14%), and both engines struggle with vowel points.
- **Fast vs Balanced:** Balanced (the app's default) was never better than
  Fast here, and on niqqud pages it was clearly worse (53.3 vs 60.1). On the
  book pages it was worse on 36 of 57 pages (−1.9 points per page, −2.8 to
  −1.1). The cause is Balanced's voting model, not its zoom-in re-reads: on
  niqqud page ws043, Balanced with its re-reads turned off still drops words
  (164 words left, against Fast's 202). Its weight against
  dropping a word only Surya read is −2.5 (Fast: −6.9), and on niqqud pages
  Tesseract misses many words, so many words are Surya-only.

## Bugs this benchmark found

Both are fixed in DualOCR (commits `71c4a7e` and `f637191`), and the results
above include the fixes. Neither fix changes any page of the author's own
evaluation set.

| Problem | Example page | Before | After |
|---|---|---|---|
| A yellowed or sepia scan counted as "highlighter" everywhere, and its cleanup wrecked the page for Tesseract | `he_wikisource` ws013 | Tesseract pass 0.8% | 71.6% |
| On small, low-resolution pages Tesseract's orientation check could call an upright page upside down, so DualOCR flipped it | `recbench_en` rb00839 | DualOCR 84.2% | 98.3% |

## How it's scored

- **Words right, in order** (Hebrew books): in-order, punctuation-free word
  accuracy against the best-matching stretch of the output. Many of these
  scans have scholarly footnotes or variant readings printed under the text
  that Wikisource doesn't transcribe. Reading them is not an error, so text
  before or after the transcribed part isn't counted. (Plain word accuracy
  over the whole output punished every system for them: on one page the
  answer key has 463 words and every engine read about 850.)
- **Surya's line score** (rec_bench): the metric from Surya's own benchmark,
  adapted to page output: each answer-key line is matched against its best
  match anywhere in the output.
- **Tests passed** (olmOCR-bench): the benchmark's own checks that a sentence
  is present, that a header or page number is absent, and that two sentences
  come in the right order. The test code is vendored in
  [`olmocr_tests/`](olmocr_tests/) from
  [allenai/olmocr](https://github.com/allenai/olmocr) (Apache-2.0); only
  its import paths were changed.
- Bracketed text is kept on both sides. DualOCR's own evaluation removes
  `[editor notes]`, but in these answer keys brackets are printed text, and
  stripping them let one stray "[" in an output delete everything up to a much
  later "]".
- Hebrew is compared with niqqud and cantillation marks removed on both
  sides.
- Left out: 3 Wikisource pages whose scan is the foreign-language original of
  a Hebrew translation (ws033, ws036, ws051), and 2 "Hebrew" rec_bench pages,
  one that is really English (rb01441) and one whose answer key is stored
  backwards (rb01442).
- The book pages were sorted into square script, Rashi script and niqqud by
  looking at the images, before any scores were read.
- No timings: other work was running on the machine during parts of the run.
  `results/scores.json` has them, with the load average next to each page.

## Datasets

None of the data is in this repository; the scripts download it.

| Name | What | Answer key | License |
|---|---|---|---|
| `he_wikisource` | Printed Hebrew book pages (1880s-1940s) from [Hebrew Wikisource](https://he.wikisource.org), proofread or validated pages, at most 3 per book | volunteer-proofread text | text CC BY-SA 4.0; scans on Wikimedia Commons, mostly public domain |
| `recbench_he`, `recbench_en` | [vikp/rec_bench](https://huggingface.co/datasets/vikp/rec_bench), Surya's own recognition benchmark (rendered pages) | text of every line | not stated |
| `funsd` | [FUNSD](https://guillaumejaume.github.io/FUNSD/) test set: 50 noisy scanned forms | every word on the form | non-commercial research and education |
| `olmocr_old_scans` | [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) `old_scans`: 1910s typed and handwritten letters | the benchmark's tests | ODC-BY 1.0 |

## Run it yourself

After `./setup.sh`, from the DualOCR folder. The full run takes a few hours
on an M4 Pro.

```sh
export DUALOCR_DATA=$PWD/bench/work      # keeps job files out of your real data folder
uv run --with pyarrow python bench/prepare_recbench.py --en 40
uv run python bench/prepare_hewikisource.py 60
uv run python bench/prepare_funsd.py
uv run python bench/prepare_olmocr.py old_scans --limit 50
bench/run_all.sh .venv/bin/python recbench_he recbench_en funsd he_wikisource olmocr_old_scans
uv run --with fuzzysearch --with beautifulsoup4 --with tqdm python bench/score.py
uv run python bench/compare.py dualocr_fast surya
```

Five systems come out of each page's single engine pass:

| System | What it is |
|---|---|
| `tesseract_raw` | `tesseract -l heb+eng --psm 3` on the page image as given |
| `tesseract` | the same on DualOCR's prepared page (orientation, deskew, highlighter cleanup): DualOCR's Tesseract pass |
| `surya` | Surya's full-page reading of the prepared page: DualOCR's Surya pass |
| `dualocr_fast` | DualOCR's alignment and voting on those two readings |
| `dualocr_balanced` | the same plus zoom-in re-reads of doubtful words |

Outputs: `results/summary.md` and `results/scores.json`, per-page text in
`results/<dataset>/pred/<system>/`, timings in
`results/<dataset>/timing.jsonl`. Engine output is cached in `work/cache/`,
so after changing DualOCR's voting, delete `results/<dataset>/pred/dualocr_*`
and run `run_all.sh` again: only the voting is redone.
