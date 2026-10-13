# Word lists

The pipeline reads one word-frequency list, `combined_wordfreq.txt`, to tell common words from rare ones (`src/lexicon.py`, rules in [docs/how-it-works.md](../../docs/how-it-works.md)). It is built from the other two files here. `filler_words.txt` derives from SCOWL (Spell Checker Oriented Word Lists, https://wordlist.aspell.net) through a Hunspell dictionary, and `scowl_wordfreq.txt` is named after it; SCOWL's notice is reproduced at the end of this file.

## `filler_words.txt`

36,653 words of 3 to 12 letters in capitals, taken from the Hunspell dictionary `en_US-large`, version 2026.02.25 (build date Wed Feb 25 15:41:02 EST 2026). Its README says that the large dictionaries correspond to SCOWL size 70 and gives the word-list command `mk-list -v1 --accents=both en_US 70`.

It was built with `scripts/build_filler_lexicon.py`, which was added in commit `ec283db` and removed in `f827bdd`. Get it back with `git show ec283db:scripts/build_filler_lexicon.py`, then:

```powershell
py -3 build_filler_lexicon.py --dic <path>\en_US-large.dic --output data\lexicon\filler_words.txt
```

The defaults are `--min-len 3 --max-len 12 --max-per-length 4000`, the values in the file's header. For each line of the `.dic` file the script keeps the word before the first `/` (the affix flags), drops every character that is not a letter, writes it in capitals, keeps lengths 3 to 12, sorts each length alphabetically and takes the first 4000 of it. Running it on that `.dic` file gives this file byte for byte.

## `scowl_wordfreq.txt`

742 words, each with a tier from 1 (rare) to 7 (most common); the header says "coarse 7-tier frequency buckets". It was added in commit `6c44eaa` (2026-02-26) and has not changed since. Its name and the docs call it SCOWL frequency tiers, but the repository does not record how the tiers were made: no script, command, SCOWL version or SCOWL list. It is kept as it is.

## `combined_wordfreq.txt`

36,794 entries, one `WORD<tab>score` per line. A word of `scowl_wordfreq.txt` keeps its tier as its score; a word only in `filler_words.txt` gets the score 0.1. Built from the repository root with the script's defaults:

```powershell
py -3 scripts/build_combined_lexicon.py
```

That is `--scowl data/lexicon/scowl_wordfreq.txt --filler data/lexicon/filler_words.txt --output data/lexicon/combined_wordfreq.txt --filler-score 0.1`. Running it on the tracked inputs gives the tracked file byte for byte.

## Source dictionary

The Hunspell dictionary folder itself is not kept in the repository, and nothing reads it: only the derived lists above are. If you rebuild `filler_words.txt`, you need that version of `en_US-large`; its README points to https://wordlist.aspell.net/dicts/.

## Copyright, sources and credits

The section "COPYRIGHT, SOURCES, and CREDITS" of `README_en_US-large.txt` in that dictionary follows word for word. The affix file `en_US-large.aff` that it also covers is not used by this repository and is not included in it.

```text
COPYRIGHT, SOURCES, and CREDITS:

The English dictionaries come directly from SCOWL and is thus under
the same copyright terms as SCOWL.  The affix file is a heavily
modified version of the original english.aff file, which was
released as part of Geoff Kuenning's Ispell and as such is covered by
his BSD license.

Copyright 2000-2026 by Kevin Atkinson

Permission to use, copy, modify, distribute, and sell any part of SCOWLv2, or
word lists created from it, is hereby granted without fee, provided that the
above copyright notice appears in all copies and that both the above
copyright notice and this notice appear in supporting documentation.  Kevin
Atkinson makes no representations about the suitability of this database for
any purpose.  It is provided "as is" without express or implied warranty.

SCOWL is derived from many sources, most of which are in the Public Domain.
Data from the Corpus of Contemporary American English (COCA) was also used.

All data from COCA comes from 3-gram data that is not freely available;
however, the usage is within the rights given by the NDA that was signed when
purchasing the data.  More information on COCA is available at
https://www.english-corpora.org/coca/.

The primary source of words for SCOWL comes from 12dicts and ENABLE2K.  Both
are in the Public Domain, but Alan Beale <biljir@pobox.com> deserves special
credit as he is the author of 12dicts and a major contributor to ENABLE2K.  In
addition, he gave me an incredible amount of feedback and created a number of
special lists in order to help improve the overall quality of SCOWL.

The initial information for Australian English comes from Benjamin Titze:

  Copyright 2016 by Benjamin Titze

  Permission to use, copy, modify, distribute and sell this array, the
  associated software, and its documentation for any purpose is hereby
  granted without fee, provided that the above copyright notice appears
  in all copies and that both that copyright notice and this permission
  notice appear in supporting documentation. Benjamin Titze makes no
  representations about the suitability of this array for any
  purpose. It is provided "as is" without express or implied warranty.

Affix file Copyright:

  Copyright 1993, Geoff Kuenning, Granada Hills, CA
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  3. All modifications to the source code must be clearly marked as
     such.  Binary redistributions based on modified source code
     must be clearly marked as modified versions in the documentation
     and/or other materials provided with the distribution.
  (clause 4 removed with permission from Geoff Kuenning)
  5. The name of Geoff Kuenning may not be used to endorse or promote
     products derived from this software without specific prior
     written permission.

  THIS SOFTWARE IS PROVIDED BY GEOFF KUENNING AND CONTRIBUTORS ``AS IS'' AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED.  IN NO EVENT SHALL GEOFF KUENNING OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
  OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
  OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
  SUCH DAMAGE.
```
