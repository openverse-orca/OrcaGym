/*
 * 按当前语言过滤 MkDocs Material 搜索结果。
 *
 * 背景：项目使用 mkdocs-static-i18n 插件，zh(默认语言)与 en 两套页面会被合并进
 * 同一份 search/search_index.json，Material 搜索本身不会按语言过滤，导致中英文
 * 页面都会搜出另一语言的结果。
 *
 * 做法：拦截搜索索引的加载（Material 通过 XMLHttpRequest 加载该 JSON），在把
 * 数据交给搜索 worker 之前，按当前语言过滤 docs，并同步修正分词语言 config.lang。
 * 这样结果数量、键盘上下键/回车跳转、搜索建议都天然正确。
 *
 * 语言判定：en 页面位于 /en/ 目录下，zh 为默认语言、位于站点根路径（无前缀）。
 * 与部署时的 base 前缀（如 /OrcaGym/）无关，本地预览与 GitHub Pages 均适用。
 */
(function () {
  "use strict";

  // 当前语言：URL 路径中是否包含 "en" 目录段
  var segments = window.location.pathname.split("/").filter(Boolean);
  var lang = segments.indexOf("en") !== -1 ? "en" : "zh";

  /**
   * 对原始搜索索引 JSON 文本按当前语言过滤，返回过滤后的 JSON 文本。
   * 失败时原样返回，保证搜索不会因此挂掉。
   */
  function filterIndex(raw) {
    try {
      var index = JSON.parse(raw);
      if (!index || !Array.isArray(index.docs)) {
        return raw;
      }

      var isEn = lang === "en";
      index.docs = index.docs.filter(function (doc) {
        var location = doc && doc.location;
        if (typeof location !== "string") {
          return true;
        }
        // en 页面 location 形如 "en/getting-started/..."，zh 页面无前缀
        var docIsEn = location.split("/").filter(Boolean)[0] === "en";
        return docIsEn === isEn;
      });

      // 只用当前语言的分词器/词干，提升中英文各自的检索效果
      if (index.config && Array.isArray(index.config.lang)) {
        index.config.lang = [lang];
      }

      return JSON.stringify(index);
    } catch (e) {
      return raw;
    }
  }

  /* ------------------------------------------------------------------ *
   * 拦截方式一：XMLHttpRequest（当前 material 9.x 用它加载搜索索引）
   * ------------------------------------------------------------------ */

  var origOpen = XMLHttpRequest.prototype.open;
  var responseTypeDesc = Object.getOwnPropertyDescriptor(
    XMLHttpRequest.prototype,
    "responseType"
  );

  // 1) 标记请求，并在 open 阶段（早于应用自身的 load 监听）挂上过滤处理
  XMLHttpRequest.prototype.open = function () {
    this.__isSearchIndex = /search\/search_index\.json$/.test(
      String(arguments[1] || "")
    );
    if (this.__isSearchIndex) {
      this.addEventListener("load", function () {
        var raw;
        try {
          raw = this.response;
        } catch (e) {
          return;
        }
        // 用过滤后的文本重建一个 Blob，替换掉应用即将读取的 response
        var blob = new Blob([filterIndex(raw)], { type: "application/json" });
        Object.defineProperty(this, "response", {
          value: blob,
          configurable: true
        });
      });
    }
    return origOpen.apply(this, arguments);
  };

  // 2) 应用会把 responseType 设为 "blob"，这里对搜索索引强制改成 "text"，
  //    这样 load 阶段才能同步读到原始 JSON 文本；随后我们仍返回 Blob，应用
  //    的 res.text() 调用不受影响。
  if (responseTypeDesc && responseTypeDesc.set) {
    Object.defineProperty(XMLHttpRequest.prototype, "responseType", {
      configurable: true,
      get: function () {
        return responseTypeDesc.get.call(this);
      },
      set: function (value) {
        if (this.__isSearchIndex && value === "blob") {
          value = "text";
        }
        return responseTypeDesc.set.call(this, value);
      }
    });
  }

  /* ------------------------------------------------------------------ *
   * 拦截方式二：fetch（未来 material 若改用 fetch 加载索引时兜底）
   * ------------------------------------------------------------------ */

  var origFetch = window.fetch;
  if (typeof origFetch === "function") {
    window.fetch = function (input, init) {
      var url =
        typeof input === "string"
          ? input
          : input && typeof input.url === "string"
            ? input.url
            : String(input);

      if (!/search\/search_index\.json$/.test(url)) {
        return origFetch.apply(this, arguments);
      }

      return origFetch.apply(this, arguments).then(function (response) {
        var clone = response.clone();
        return response
          .json()
          .then(function (index) {
            return new Response(filterIndex(JSON.stringify(index)), {
              status: 200,
              headers: { "Content-Type": "application/json" }
            });
          })
          .catch(function () {
            return clone;
          });
      });
    };
  }
})();
