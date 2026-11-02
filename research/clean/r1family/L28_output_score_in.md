# Detection scoring (L28_output)

For each unit: read the description. Then pick EXACTLY 5 of the 10 items
where you think the unit is active. Items are text contexts (the active token is in [[ ]]) or words.


## U001

Description: Pushes the next token toward closing punctuation with parenthesis or quote, pipe symbols and control characters, plus stray fragments; mixed.
Items:
1. 'onics'
2. 'mu'
3. 'recht'
4. '”),'
5. ' merit'
6. 'ipl'
7. ' Luckily'
8. '…..'
9. ' perspect'
10. ' impacted'

## U010

Description: Pushes the next token toward stems of long words (e.g. ' affili', ' strugg', ' acknow', ' purpos', ' encourag') plus a few whole words.
Items:
1. ' acquaint'
2. ' christ'
3. ' folders'
4. ' unheard'
5. 'Academ'
6. ' clf'
7. ' MOD'
8. 'Interestingly'
9. ' orders'
10. ' incons'

## U015

Description: Pushes the next token toward odd formatting tokens: pipe-table symbols, control characters, 'reponame' and repeated punctuation; fires after units like cm or mm.
Items:
1. '?]('
2. '“.'
3. ' session'
4. ' wheat'
5. 'faiss'
6. '|"\''
7. ' blossoms'
8. "|':'"
9. ' band'
10. ' represented'

## U022

Description: Pushes the next token toward unusual formatting tokens: pipe-separated table symbols, control or whitespace characters, and quote-plus-punctuation endings.
Items:
1. ' yards'
2. ' inv'
3. ' buff'
4. '�'
5. '”?'
6. '’).'
7. '”),'
8. '%|'
9. ' upheaval'
10. ' investigations'

## U027

Description: Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
Items:
1. 'Random'
2. ' doub'
3. 'ipelago'
4. 'rottle'
5. 'ymers'
6. 'published'
7. ' opposites'
8. '"\''
9. 'billed'
10. ' chimpan'

## U036

Description: Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
Items:
1. 'published'
2. ' acquainted'
3. 'Random'
4. 'ipelago'
5. ' utilizing'
6. 'billed'
7. ' Technique'
8. ' belonged'
9. ' chimpan'
10. ' Illustrated'

## U041

Description: Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
Items:
1. ' journalist'
2. ' surfact'
3. 'Random'
4. 'ipelago'
5. 'published'
6. 'Requ'
7. 'billed'
8. 'etheless'
9. ' pair'
10. ' chimpan'

## U046

Description: Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
Items:
1. 'Random'
2. ' chimpan'
3. 'hake'
4. 'billed'
5. 'published'
6. ' Historian'
7. 'ipelago'
8. "');"
9. ' ds'
10. ' Karen'
