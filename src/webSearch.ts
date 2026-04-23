export interface WebSearchResult {
  title: string;
  url: string;
  snippet: string;
  source: string;
}

export interface WebSearchResponse {
  query: string;
  provider: string;
  results: WebSearchResult[];
}

type DuckDuckGoTopic = {
  Text?: string;
  FirstURL?: string;
  Name?: string;
  Topics?: DuckDuckGoTopic[];
};

type DuckDuckGoResponse = {
  Heading?: string;
  AbstractText?: string;
  AbstractURL?: string;
  Results?: DuckDuckGoTopic[];
  RelatedTopics?: DuckDuckGoTopic[];
};

type WikipediaSearchResponse = {
  query?: {
    search?: Array<{
      title?: string;
      snippet?: string;
    }>;
  };
};

type SearxngResponse = {
  query?: string;
  results?: Array<{
    title?: string;
    url?: string;
    content?: string;
    engine?: string;
  }>;
};

function cleanText(value: string | undefined): string {
  return (value ?? '').replace(/\s+/g, ' ').trim();
}

function sourceFromUrl(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return 'web';
  }
}

function splitTopicText(text: string): { title: string; snippet: string } {
  const cleaned = cleanText(text);
  const separator = cleaned.indexOf(' - ');
  if (separator <= 0) return { title: cleaned.slice(0, 80), snippet: cleaned };
  return {
    title: cleaned.slice(0, separator),
    snippet: cleaned.slice(separator + 3),
  };
}

function stripHtmlSnippet(value: string | undefined): string {
  const html = value ?? '';
  if (typeof DOMParser === 'undefined') return cleanText(html.replace(/<[^>]*>/g, ' '));
  const doc = new DOMParser().parseFromString(html, 'text/html');
  return cleanText(doc.documentElement.textContent ?? '');
}

function addResult(
  results: WebSearchResult[],
  seenUrls: Set<string>,
  title: string | undefined,
  url: string | undefined,
  snippet: string | undefined,
  limit: number,
) {
  const cleanUrl = cleanText(url);
  const cleanSnippet = cleanText(snippet);
  if (!cleanUrl || !cleanSnippet || seenUrls.has(cleanUrl) || results.length >= limit) return;

  seenUrls.add(cleanUrl);
  results.push({
    title: cleanText(title) || sourceFromUrl(cleanUrl),
    url: cleanUrl,
    snippet: cleanSnippet,
    source: sourceFromUrl(cleanUrl),
  });
}

function collectTopics(
  topics: DuckDuckGoTopic[] | undefined,
  results: WebSearchResult[],
  seenUrls: Set<string>,
  limit: number,
) {
  for (const topic of topics ?? []) {
    if (results.length >= limit) return;
    if (topic.Topics?.length) {
      collectTopics(topic.Topics, results, seenUrls, limit);
      continue;
    }

    const { title, snippet } = splitTopicText(topic.Text ?? '');
    addResult(results, seenUrls, title, topic.FirstURL, snippet, limit);
  }
}

async function searchDuckDuckGo(
  query: string,
  results: WebSearchResult[],
  seenUrls: Set<string>,
  limit: number,
  signal?: AbortSignal,
) {
  const endpoint = new URL('https://api.duckduckgo.com/');
  endpoint.searchParams.set('q', query);
  endpoint.searchParams.set('format', 'json');
  endpoint.searchParams.set('no_html', '1');
  endpoint.searchParams.set('no_redirect', '1');
  endpoint.searchParams.set('skip_disambig', '1');

  const response = await fetch(endpoint, {
    signal,
    headers: { Accept: 'application/json' },
  });

  if (!response.ok) {
    throw new Error(`DuckDuckGo failed with HTTP ${response.status}`);
  }

  const data = (await response.json()) as DuckDuckGoResponse;

  addResult(
    results,
    seenUrls,
    data.Heading,
    data.AbstractURL,
    data.AbstractText,
    limit,
  );
  collectTopics(data.Results, results, seenUrls, limit);
  collectTopics(data.RelatedTopics, results, seenUrls, limit);
}

async function searchSearxng(
  baseUrl: string,
  query: string,
  results: WebSearchResult[],
  seenUrls: Set<string>,
  limit: number,
  signal?: AbortSignal,
) {
  const endpoint = new URL(`${baseUrl.replace(/\/+$/, '')}/search`);
  endpoint.searchParams.set('q', query);
  endpoint.searchParams.set('format', 'json');
  endpoint.searchParams.set('safesearch', '0');

  const response = await fetch(endpoint, {
    signal,
    headers: { Accept: 'application/json' },
  });

  if (!response.ok) {
    throw new Error(`SearxNG failed with HTTP ${response.status}`);
  }

  const data = (await response.json()) as SearxngResponse;
  for (const item of data.results ?? []) {
    if (results.length >= limit) return;
    const title = cleanText(item.title);
    const url = cleanText(item.url);
    const snippet = cleanText(item.content);
    if (!url || !snippet) continue;

    const cleanedUrl = url;
    if (seenUrls.has(cleanedUrl)) continue;
    seenUrls.add(cleanedUrl);
    results.push({
      title: title || sourceFromUrl(cleanedUrl),
      url: cleanedUrl,
      snippet,
      source: cleanText(item.engine) || sourceFromUrl(cleanedUrl),
    });
  }
}

async function searchWikipedia(
  query: string,
  results: WebSearchResult[],
  seenUrls: Set<string>,
  limit: number,
  signal?: AbortSignal,
) {
  if (results.length >= limit) return;

  const endpoint = new URL('https://en.wikipedia.org/w/api.php');
  endpoint.searchParams.set('action', 'query');
  endpoint.searchParams.set('list', 'search');
  endpoint.searchParams.set('srsearch', query);
  endpoint.searchParams.set('srlimit', String(limit - results.length));
  endpoint.searchParams.set('format', 'json');
  endpoint.searchParams.set('origin', '*');

  const response = await fetch(endpoint, {
    signal,
    headers: { Accept: 'application/json' },
  });

  if (!response.ok) {
    throw new Error(`Wikipedia failed with HTTP ${response.status}`);
  }

  const data = (await response.json()) as WikipediaSearchResponse;
  for (const item of data.query?.search ?? []) {
    const title = cleanText(item.title);
    if (!title) continue;
    const url = `https://en.wikipedia.org/wiki/${encodeURIComponent(title.replace(/\s+/g, '_'))}`;
    addResult(results, seenUrls, title, url, stripHtmlSnippet(item.snippet), limit);
  }
}

export interface WebSearchOptions {
  signal?: AbortSignal;
  limit?: number;
  searxngUrl?: string;
}

export async function webSearch(
  query: string,
  options: WebSearchOptions = {},
): Promise<WebSearchResponse> {
  const limit = Math.min(8, Math.max(1, options.limit ?? 5));
  const results: WebSearchResult[] = [];
  const seenUrls = new Set<string>();
  const errors: string[] = [];

  const searxngUrl = options.searxngUrl?.trim();
  const providers: string[] = [];

  if (searxngUrl) {
    try {
      await searchSearxng(searxngUrl, query, results, seenUrls, limit, options.signal);
      providers.push(`SearxNG (${sourceFromUrl(searxngUrl)})`);
    } catch (error) {
      if (options.signal?.aborted) throw error;
      errors.push(error instanceof Error ? error.message : String(error));
    }
  }

  if (results.length < limit) {
    try {
      await searchDuckDuckGo(query, results, seenUrls, limit, options.signal);
      providers.push('DuckDuckGo Instant Answer');
    } catch (error) {
      if (options.signal?.aborted) throw error;
      errors.push(error instanceof Error ? error.message : String(error));
    }
  }

  if (results.length < limit) {
    try {
      await searchWikipedia(query, results, seenUrls, limit, options.signal);
      providers.push('Wikipedia');
    } catch (error) {
      if (options.signal?.aborted) throw error;
      errors.push(error instanceof Error ? error.message : String(error));
    }
  }

  if (!results.length && errors.length) {
    throw new Error(errors.join('; '));
  }

  return {
    query,
    provider: providers.length ? providers.join(' + ') : 'none',
    results,
  };
}

export function formatWebSearchContext(search: WebSearchResponse, userMessage: string): string {
  const resultLines = search.results.length
    ? search.results
      .map((result, index) => {
        const n = index + 1;
        return [
          `[${n}] ${result.title}`,
          `Source: ${result.source}`,
          `URL: ${result.url}`,
          `Snippet: ${result.snippet}`,
        ].join('\n');
      })
      .join('\n\n')
    : 'No useful results were returned.';

  return [
    'Answer the latest user message. Web search was enabled for this turn.',
    'Use the web results below when they are relevant. If the results are weak or unrelated, say that plainly.',
    'When using a result, cite it inline with bracketed source numbers like [1] or [2].',
    '',
    'Latest user message:',
    userMessage,
    '',
    `Web search provider: ${search.provider}`,
    `Web search query: ${search.query}`,
    '',
    'Web search results:',
    resultLines,
  ].join('\n');
}
