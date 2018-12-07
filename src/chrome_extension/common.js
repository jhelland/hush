/**
 * Copyright 2017 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Common utility functions.
 */
const Utils = class {
  /**
   * Loads template from a URL.
   * @param {string} url The url of the template.
   * @return {Promise} Promise containing the text within the template file.
   */
  loadTemplate(url) {
    return new Promise((resolve, reject) => {
      const x = new XMLHttpRequest();
      x.open('GET', url, true);
      x.onreadystatechange = () => {
        if (x.readyState == 4 && x.status == 200) {
          resolve(x.responseText);
        } else if (x.status == 400) {
          reject();
        }
      };
      x.onerror = reject;
      x.send(null);
    });
  }

  /**
   * Renders Mustache template and inserts the element relative to a reference
   * element.
   * @param {string} templateUrl The filepath of the Mustache template.
   * @param {Object} templateParams The parameters to pass into the template.
   * @param {Element} referenceElement The element of reference to insert the
   *     object into or before.
   * @param {Function=} opt_onFulfilled The optional function to execute after
   *     inserting the template.
   * @param {boolean} [append=true] Whether or not to append the element.
   */
  insertTemplate(templateUrl, templateParams, referenceElement, opt_onFulfilled,
      append = true) {
    this.loadTemplate(templateUrl).then((template) => {
      const renderedTemplate = Mustache.render(template, templateParams);
      const renderedElement =
          document.createRange().createContextualFragment(renderedTemplate);
      if (append) {
        referenceElement.appendChild(renderedElement);
      } else {
        referenceElement.parentNode.insertBefore(renderedElement,
          referenceElement);
      }
    }).then(() => {
      if (opt_onFulfilled) {
        opt_onFulfilled();
      }
    });
  }

  /**
   * Creates a DOM element.
   * @param {string} tagName
   * @param {string} className
   * @param {string} text
   * @returns {Element}
   */
  createElement(tagName, className, text) {
    const element = document.createElement(tagName);
    element.classList.add(className);
    element.innerText = text;
    return element;
  }
};

/**
 * Copyright 2017 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

const PubSubEvent = {
  INFO_PANEL_READY: 'infoPanelReady',
  COMMENT_FILTER_READY: 'commentFilterReady',
};

/**
 * Publishes and subscribes to events on the page.
 */
const PubSub = class {
  constructor() {
    /** @private {Object} The topics for subscription. */
    this.topics_ = {};
  }

  /**
   * Subscribes to a topic.
   * @param {string} topic The topic for subscription.
   * @param {Function} listener The listener subscribing.
   * @param {Object=} opt_scope The optional scope.
   * @return {Object}
   */
  subscribe(topic, listener, opt_scope) {
    // Creates the topic's object if not yet created.
    if (!this.topics_[topic]) {
      this.topics_[topic] = [];
    }

    // Constructs listener object with scope if defined.
    let listenerObject = {'listener': listener};
    if (opt_scope) {
      listenerObject.scope = opt_scope;
    }

    // Adds the listener to queue.
    const index = this.topics_[topic].push(listenerObject) - 1;

    // Provides handle back for removal of topic.
    return {
      remove: function() {
        delete this.topics_[topic][index];
      },
    };
  }

  /**
   * Publishes to a topic.
   * @param {string} topic The topic in subscription.
   * @return {undefined}
   */
  publish(topic, ...args) {
    // Exits if the topic doesn't exist, or there's no listeners in queue.
    if (!this.topics_[topic]) {
      return;
    }

    // Cycles through topics queue.
    for (let item of this.topics_[topic]) {
      if (item.scope) {
        item.listener.call(item.scope, args);
      } else {
        item.listener(args);
      }
    }
  }
};

const pubSub = new PubSub();
/**
 * Copyright 2017 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * CHANGES:
 *
 * Removed several websites.
 * Added 4chan.
 */
const CommentSelectors = {
  'twitter.com': {
    'ID': '#timeline',
    'COMMENTS_CONTAINER': '.tweet',
    'COMMENT_WRAPPER': 'content',
    'COMMENT_CONTENT_CONTAINER': 'js-tweet-text-container',
    'COMMENT_CONTENT': 'tweet-text',
    'NEW_COMMENT_MUTATION': 'tweet-text',
  },
  /*'reddit.com': {
    'ID': 'sogqxs-0 jlPabA',
    'COMMENTS_CONTAINER': 'Comment t1_eb2qbwz',
    'COMMENT_WRAPPER': 's1ook3io-2 emJXdb',
    'COMMENT_CONTENT_WRAPPER': '_3h7_WaVl17ooQTHO2Uzr2s s1hmcfrd-0 ckueCN',
    'COMMENT_CONTENT': 'yklcuq-10 hpxQMr',
    'NEW_COMMENT_MUTATION': '',
  },*/
  'nextdoor.com': {
    'ID': 'feed-container',
    'COMMENTS_CONTAINER': '.feed-container',  // note: '.' suffix, without this, target isn't a valid node when passed to observe() for MutationObserver
    'COMMENT_WRAPPER': 'ui-container',
    'COMMENT_CONTENT_WRAPPER': 'content-body',
    'COMMENT_CONTENT': 'Linkify',
    'NEW_COMMENT_MUTATION': 'post-container',
  },
  'boards.4channel.org': {
    'ID': 'board',
    'COMMENTS_CONTAINER': '.postContainer',
    'COMMENT_WRAPPER': 'postContainer',
    'COMMENT_CONTENT_WRAPPER': 'desktop',
    'COMMENT_CONTENT': 'postMessage',
    'NEW_COMMENT_MUTATION': 'postMessage',
  },
  'boards.4chan.org': {
    'ID': 'board',
    'COMMENTS_CONTAINER': 'postContainer',
    'COMMENT_WRAPPER': 'postContainer',
    'COMMENT_CONTENT_WRAPPER': 'desktop',
    'COMMENT_CONTENT': 'postMessage',
  },
  'www.telegraph.co.uk': {
    'ID': 'comments',
    'COMMENTS_CONTAINER': '#comments',
    'COMMENT_DATE': 'meta[itemprop=dateCreated',
    'COMMENT_WRAPPER': 'fyre-comment-wrapper',
    'COMMENT_CONTENT_WRAPPER': 'fyre-comment-body',
    'COMMENT_CONTENT': 'fyre-comment',
    'MORE_LINK': 'comments-block__load',
    'NEW_COMMENT_MUTATION': 'fyre-comment-article',
  },
};


/**
 * Scrapes webpage for potential comments and stores their respecitve
 * toxicity rating.
 */
const Comments = class {
  constructor() {
    /** @type {number} The count of toxic comments. */
    this.toxicCommentCount = 0;
    /** @type {Array<Element>} The array of comments. */
    this.commentStream;
  }

  /**
   * Gets a stream of comments identified by the given class.
   * @param {string} commentClass The class name of the comments to get.
   * @return {Array<Element>} The array of comment elements.
   */
  getComments(commentClass) {
    this.commentStream = document.getElementsByClassName(commentClass);
    return this.commentStream;
  }

  /**
   * Gets the comment selector names for each publisher.
   * @return {Object} The object defining selector names indexed by publisher.
   */
  getSelectors() {
    return CommentSelectors;
  }
};

/**
 * Copyright 2017 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/**
 * Uses localhost server API to analyze toxicity of a given comment.
 */
const ToxicityAnalyzer = class {
  /**
   * Gets the parsed Date from a string representation.
   * @param {string} commentDate String representing the comment date.
   * @return {Date} The parsed Date.
   * @private
   */
  getParsedDate_(commentDate) {
    if (commentDate && commentDate.indexOf('ago') >= 0) {
      commentDate = commentDate.replace('ago', '');
      // This is capturing the 'less than one minute ago' datestamp.
      if (commentDate === null) {
        commentDate = new Date();
      }
      return Date.parse('-' + commentDate);
    } else {
      return new Date(commentDate);
    }
  }

  /**
   * Analyzes the given comment.
   * @param {string} comment The comment content to parse and analyze.
   * @param {string=} opt_date The optional comment date.
   * @return {Promise} Promise containing the comment, toxicity analysis, and
   *   parsed date.
   */
  analyze(comment, opt_date) {
    const parsedDate = this.getParsedDate_(opt_date);
    const parsedComment = comment.replace(/"/g, "");
    return new Promise((resolve, reject) => {
      // Perspective API doesn't accept comments longer than 3000 characters.
      if (!comment || comment.length >= 3000) {
        reject();
        return false;
      }
      const analyzeURL = 'http://localhost:5000/api'
      const x = new XMLHttpRequest();
      const composedComment = `{query: "${parsedComment}"}`;
      // Adds the required HTTP header for form data POST requests.
      x.open('POST', analyzeURL, true);
      x.setRequestHeader('Content-Type', 'application/json');
      x.withCredentials = false;
      // The localhost server responds with JSON, so let Chrome parse it.
      x.responseType = 'json';
      x.onreadystatechange = () => {
        if (x.readyState == 4 && x.status == 200) {
          resolve({
            comment: comment,
            response: x.response,
            date: parsedDate
          })
        } else if (x.status == 400) {
          reject();
        }
      };
      x.onerror = reject;
      x.send("comment=" + composedComment); // send comment in raw binary
    });
  }

  /**
   * Gets the toxicity of the given localhost server response
   * @param {Object} response The response from the Perspective API analysis.
   * @return {?number} The toxicity score if it exists.
   */
  getToxicity(response) {
    return response.score || null;
  }

  /**
   * Analyzes the given comment and returns only the toxicity.
   * @param {string} comment The comment content to parse and analyze.
   * @return {Promise} Promise containing toxicity score.
   */
  getCommentToxicity(comment) {
    return new Promise((resolve, reject) => {
      this.analyze(comment).then((response) => {
        resolve(this.getToxicity(response.response));
      });
    });
  }
};


/**
 * Copyright 2017 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * The JSON configuration of the mutation observer.
 * @readonly
 * @enum {boolean}
 */
const COMMENTWATCHER_CONFIG = {
  attributes: false,
  childList: true,
  characterData: false,
  subtree: true,
};

/**
 * Watches for mutations to the comments container and analyzes when new
 *   comments are available.
 */
const CommentWatcher = class {
  constructor(analyzeComments) {
    /** @private {Function} The function to analyze comments. */
    this.analyzeComments_ = analyzeComments;
    /** @private {number} The total number of comments. */
    this.totalComments_ = 0;
    /** @private {Function} The function to get comments from the page. */
    this.getComments_ = () =>
        comments.getComments(CommentSelectors[hostname].COMMENT_WRAPPER);

    this.setupMutationCache_();
    this.initMutationObserver_();
  }


  /**
   * Intializes the mutation observer and observes the target.
   * @private
   */
  initMutationObserver_() {
    // The element of the comments selector.
    const target = document.querySelectorAll(
      CommentSelectors[hostname].COMMENTS_CONTAINER)[0];

    /** @private {MutationObserver} The mutation observer. */
    this.observer_ = new MutationObserver((mutations) => {
      // Runs the cached functions once all comments have been loaded in.
      for (let mutation of mutations) {
        if (this.isCommentNode_(mutation)) {
          this.startMutationCache_();
        }
      }
    });

    // Passes in the target node and the observer options.
    this.observer_.observe(target, COMMENTWATCHER_CONFIG);
  }

  /**
   * Determines whether the mutation is performed on a comment node.
   * @return {boolean} Whether the mutated node is a comment.
   */
  isCommentNode_(mutation) {
    const containsCommentArticle = (elements) => {
      for (let element of elements) {
        if (element.classList && (element.classList.contains(
            CommentSelectors[hostname].NEW_COMMENT_MUTATION))) {
          return true;
        }
      }
      return false;
    };

    return (mutation.target.id == CommentSelectors[hostname].ID) ||
        containsCommentArticle(mutation.addedNodes);
  }

  /**
   * Sets up function to analyze new comments to be cached for later execution.
   * @private
   */
  setupMutationCache_() {
    this.mutationCache_ = () => {
      let newComments = this.getComments_().length;
      if (newComments > this.totalComments_) {
        this.analyzeComments_(this.getComments_());
      }
      this.totalComments_ = newComments;
    };
  }

  /**
   * Executes cached functions every 1000 ms.
   * @private
   */
  startMutationCache_() {
    setInterval(this.mutationCache_, 1000);
  }
};
/**
 * Copyright 2017 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

const utils = new Utils();
const infoPanelTemplate = 'templates/info-panel.mst';

/**
 * Class names contained within the InfoPanel component.
 * @readonly
 * @enum {string}
 */
const InfoPanelClass = {
  CLOSE_ICON: 'info-panel__close-icon',
  DETAILS: 'info-panel__details',
  DETAILS_HIDDEN: 'info-panel__details--hidden',
  DETAILS_VISIBLE: 'info-panel__details--visible',
  HIDDEN: 'info-panel--hidden',
  INFO_PANEL: 'info-panel',
  INFO_TOGGLE: 'info-toggle',
  INNER: 'info-panel__inner',
  COMMENT_FILTERED: 'comment-filtered',
};

const InfoPanelIcon = {
  DEFAULT: {
    CLOSE: 'close.svg',
    DOWN: 'down.svg',
    DRAG: 'drag-arrow-white.svg',
  },
  WHITE: {
    CLOSE: 'close-white.svg',
    DOWN: 'down-white.svg',
    DRAG: 'drag-arrow-white.svg',
  }
}
/**
 * Hides the InfoPanel from the page with CSS classes.
 */
const hideInfoPanel = (element) => {
  element.classList.add(InfoPanelClass.HIDDEN);
  document.body.classList.remove(InfoPanelClass.COMMENT_FILTERED);
};

/**
 * Defines the InfoPanel overlay to be added to the page.
 */
const InfoPanel = class {
  constructor(title = '', subtitle, description, containerElement,
      closeFunction = hideInfoPanel, isDarkTheme) {
    /** @private {string} The title of the extension. */
    this.title_ = title;
    /** @private {string} The title of the extension. */
    this.subtitle_ = subtitle;
    /** @private {string} The description of the extension. */
    this.description_ = description;
    /** @private {Element} The element to contain the InfoPanel. */
    this.containerElement_ = containerElement;
    /** @private {string} The function to close the InfoPanel. */
    this.closeFunction_ = closeFunction;
    /** @private {string} The InfoPanel element to be assigned on setup. */
    this.infoPanelElement_;
    /** @private {id} The id by which to identify the InfoPanel element. */
    this.id_ = InfoPanelClass.INFO_PANEL;
    /** @private {string} The icon set to use in the InfoPanel */
    this.icon_ = (isDarkTheme ? InfoPanelIcon.WHITE : InfoPanelIcon.DEFAULT);

    this.setupInfoPanel_();
  }

  /**
   * Sets up the InfoPanel by adding it to the page with details toggle and
   * close listeners.
   * @private
   */
  setupInfoPanel_() {
    const templateUrl = chrome.runtime.getURL(infoPanelTemplate);
    const templateParams = {
      closeIcon: chrome.runtime.getURL(this.icon_.CLOSE),
      downArrow: chrome.runtime.getURL(this.icon_.DOWN),
      dragArrow: chrome.runtime.getURL(this.icon_.DRAG),
      title: this.title_,
      subtitle: this.subtitle_,
      description: this.description_,
    };
    utils.insertTemplate(templateUrl, templateParams, this.containerElement_,
        () => {
      this.infoPanelElement_ = document.getElementById(this.id_);
      pubSub.publish(PubSubEvent.INFO_PANEL_READY);

      // Toggles additional information when user clicks the toggle icon.
      const infoToggle = document.getElementById(InfoPanelClass.INFO_TOGGLE);
      infoToggle.addEventListener('click', () => {
        this.infoPanelElement_.classList.toggle(InfoPanelClass.DETAILS_VISIBLE);
      });

      // Closes the InfoPanel when user clicks the close icon.
      const closeLink =
          document.getElementsByClassName(InfoPanelClass.CLOSE_ICON)[0];
      closeLink.addEventListener('click', () => {
        this.closeFunction_(this.infoPanelElement_);
      });
    });
  }
};
