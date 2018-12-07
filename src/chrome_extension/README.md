Code by [raisaveuc](https://github.com/raisaveuc), [gurmukhp](https://github.com/gurmukhp), [bradleyg](https://github.com/bradleyg)
Adapted by [Jonathan Helland]()

## Comment Blur Filter
Easily find and hide comments based on your tolerance for toxicity.

### Background
Comments are scraped from the comment-enabled article in view and then scored using Perspective API's AnalyzeComment request. This data is then shown as a graph of toxicity over time. The idea is that it may be possible to identify trends in the toxicity of comments.

### Usage
Ensure that the localhost server is running and accepting requests. Then, visit any article that contains comments from the below publishers. Alternatively, visit any 4chan board. Once loaded, click the app icon in the extensions panel to see comments filtered by the default threshold. By sliding the control, the comments in the story will be hidden or shown dependent on the desired toxicity score. On 4chan, thumbnails associated with toxic posts will be blurred.

### Adding a new website
To add a new website, the following files must be updated: 
- common.js
- manifest.json
- comment-filter.js.

Websites available:
- [Telegraph](https://www.telegraph.co.uk)
- [4chan](https://www.4chan.org)
- [Nextdoor](https://nextdoor.com)

Blacklist:
- Twitter: blocks scraping 
- Facebook: requests to API violate security policy
