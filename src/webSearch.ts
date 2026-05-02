export interface WebSearchResult {
  title: string;
  url: string;
  snippet: string;
  source: string;
}

export interface WebSearchFact {
  title: string;
  value: string;
  source: string;
}

export interface WebSearchResponse {
  query: string;
  provider: string;
  facts: WebSearchFact[];
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
  answers?: string[];
  infoboxes?: Array<{
    infobox?: string;
    content?: string;
    attributes?: Array<{
      label?: string;
      value?: string;
    }>;
  }>;
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

function addFact(facts: WebSearchFact[], title: string, value: string | undefined, source: string) {
  const cleanValue = cleanText(value);
  if (!cleanValue) return;
  if (facts.some((fact) => fact.title === title && fact.value === cleanValue)) return;
  facts.push({
    title,
    value: cleanValue,
    source,
  });
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
  facts: WebSearchFact[],
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
  for (const answer of data.answers ?? []) {
    addFact(facts, 'SearxNG answer', answer, 'SearxNG');
  }
  for (const infobox of data.infoboxes ?? []) {
    addFact(facts, cleanText(infobox.infobox) || 'SearxNG infobox', infobox.content, 'SearxNG');
    for (const attribute of infobox.attributes ?? []) {
      addFact(
        facts,
        cleanText(attribute.label) || cleanText(infobox.infobox) || 'SearxNG infobox',
        attribute.value,
        'SearxNG',
      );
    }
  }

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

const TIME_QUERY_PATTERNS = [
  /\bwhat\s+(?:is|'s)\s+the\s+current\s+time\s+in\s+([a-z][a-z\s._/-]{1,60})\??$/i,
  /\bcurrent\s+time\s+in\s+([a-z][a-z\s._/-]{1,60})\??$/i,
  /\bwhat\s+time\s+is\s+it\s+in\s+([a-z][a-z\s._/-]{1,60})\??$/i,
  /\btime\s+(?:is\s+it\s+)?(?:now\s+)?(?:in|at)\s+([a-z][a-z\s._/-]{1,60})\??$/i,
];

function normalizePlace(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/\b(?:right\s+now|now|today|please)\b/g, '')
    .replace(/[^a-z]+/g, ' ')
    .trim();
}

function timeZoneCityName(timeZone: string): string {
  return timeZone.split('/').pop()?.replace(/_/g, ' ').toLowerCase() ?? '';
}

function timeZones(): string[] {
  const intlWithZones = Intl as typeof Intl & {
    supportedValuesOf?: (key: 'timeZone') => string[];
  };
  return intlWithZones.supportedValuesOf?.('timeZone') ?? [
    'Europe/Istanbul',
    'Europe/London',
    'Europe/Paris',
    'Europe/Berlin',
    'America/New_York',
    'America/Los_Angeles',
    'Asia/Tokyo',
    'Asia/Dubai',
  ];
}

function findTimeZoneForQuery(query: string): string | null {
  const match = TIME_QUERY_PATTERNS.map((pattern) => query.match(pattern)).find(Boolean);
  if (!match?.[1]) return null;

  const place = normalizePlace(match[1]);
  if (!place) return null;

  const zones = timeZones();
  return zones.find((zone) => timeZoneCityName(zone) === place)
    ?? zones.find((zone) => timeZoneCityName(zone).includes(place))
    ?? null;
}

function formatCurrentTimeFact(query: string): WebSearchFact | null {
  const timeZone = findTimeZoneForQuery(query);
  if (!timeZone) return null;

  const now = new Date();
  const formatted = new Intl.DateTimeFormat(undefined, {
    dateStyle: 'full',
    timeStyle: 'long',
    timeZone,
  }).format(now);

  return {
    title: `Current time (${timeZone})`,
    value: formatted,
    source: 'Browser Intl clock',
  };
}

export async function webSearch(
  query: string,
  options: WebSearchOptions = {},
): Promise<WebSearchResponse> {
  const limit = Math.min(8, Math.max(1, options.limit ?? 5));
  const facts: WebSearchFact[] = [];
  const results: WebSearchResult[] = [];
  const seenUrls = new Set<string>();
  const errors: string[] = [];

  const searxngUrl = options.searxngUrl?.trim();
  const providers: string[] = [];
  const timeFact = formatCurrentTimeFact(query);
  if (timeFact) facts.push(timeFact);

  if (searxngUrl) {
    try {
      await searchSearxng(searxngUrl, query, facts, results, seenUrls, limit, options.signal);
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
    facts,
    results,
  };
}

export function formatWebSearchContext(search: WebSearchResponse, userMessage: string): string {
  const factLines = search.facts.length
    ? search.facts
      .map((fact, index) => {
        const n = index + 1;
        return [
          `[F${n}] ${fact.title}`,
          `Source: ${fact.source}`,
          `Value: ${fact.value}`,
        ].join('\n');
      })
      .join('\n\n')
    : 'No direct answers or computed facts were found.';

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
    'First use any direct answers or computed facts below. Use web result snippets only as supporting context.',
    'Use the web results below when they are relevant. If the results are weak or unrelated, say that plainly.',
    'When using a result, cite it inline with bracketed source numbers like [1] or [2].',
    'When using a direct fact, cite it inline like [F1].',
    '',
    'Latest user message:',
    userMessage,
    '',
    `Web search provider: ${search.provider}`,
    `Web search query: ${search.query}`,
    '',
    'Direct answers and computed facts:',
    factLines,
    '',
    'Web search results:',
    resultLines,
  ].join('\n');
}
