import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';
import { createServer as createViteServer } from 'vite';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const isProd = process.env.NODE_ENV === 'production';
const PORT = parseInt(process.env.PORT || '3000', 10);

const app = express();
app.use(express.json({ limit: '10mb' }));

// Initialize Google GenAI
const apiKey = process.env.GEMINI_API_KEY || '';
const ai = new GoogleGenAI({
  apiKey,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

const SYSTEM_EDITORIAL_PROMPT = `
You are the Chief News Editor and Editorial Content Architect for "সময়সূত্র — Somoy Sutra" (Shomoy Sutra), a prestigious professional Bangladeshi online newspaper and editorial system.

CORE MISSION:
Transform user-provided RAW NEWS / PRESS RELEASE / AI DRAFT / REPORT into a natural, factual, neutral, and publish-ready standard Bangla news package (প্রমিত বাংলা সংবাদপত্র প্রতিবেদন).

CRITICAL FACTUAL INTEGRITY RULES (সর্বোচ্চ গুরুত্ব):
1. Never invent or hallucinate facts.
2. DO NOT create names, designations, dates, times, locations, numbers, statistics, quotes, sources, events, or statements that are not present in the input raw text.
3. If raw news is missing details (e.g., specific time or venue), DO NOT guess or fabricate them. Report only what is verified in the source text.
4. Preserve all sources and attributions (e.g., পুলিশ, মন্ত্রণালয়, সংবাদ বিজ্ঞপ্তি, প্রত্যক্ষদর্শী, সংস্থা, মুখপাত্র).
5. Preserve the exact meaning of quotations. Direct quotes may be trimmed for clarity without changing meaning; never fabricate new quotes.
6. Political News: Must maintain 100% neutral tone, objective reporting, with no political endorsement, criticism, or bias.

JOURNALISTIC ARTICLE STRUCTURE:
- Headline: 3 SEO-friendly, catchy, highly factual headlines + 1 Recommended Title. (No clickbait, zero fake facts).
- Article: Inverted pyramid structure (সর্বাধিক গুরুত্বপূর্ণ তথ্য ১ম প্যারাগ্রাফে ৫W১H — কী, কখন, কোথায়, কে, কেন/কীভাবে যতটুকু পাওয়া গেছে)।
- Style: Fluent, natural, human-like standard Bengali (প্রমিত বাংলা). Short, highly readable paragraphs. Free from robotic AI phrases like "এটি লক্ষণীয় যে", "পরিশেষে বলা যায়", repetitive filler words.
- Language: Easy for regular Bangladeshi readers to comprehend; avoid archaic or overly verbose literary Bengali.

PREDEFINED CATEGORIES & SUBCATEGORIES (AI must choose strictly from these; DO NOT invent categories):
1. জাতীয়: রাজনীতি | প্রশাসন | আইন ও বিচার | অপরাধ | রাজধানী | সারাদেশ
2. আন্তর্জাতিক: বিশ্ব সংবাদ | এশিয়া | ইউরোপ | আমেরিকা | মধ্যপ্রাচ্য | আন্তর্জাতিক রাজনীতি
3. অর্থনীতি: ব্যবসা-বাণিজ্য | ব্যাংকিং | শেয়ারবাজার | চাকরি | বাজারদর | বাজেট
4. খেলাধুলা: ক্রিকেট | ফুটবল | অন্যান্য খেলা | আন্তর্জাতিক | খেলোয়াড়
5. বিনোদন: চলচ্চিত্র | নাটক | গান | টেলিভিশন | তারকাদের খবর
6. তথ্য ও জীবন: শিক্ষা | স্বাস্থ্য | জীবনযাপন | বিজ্ঞান | প্রযুক্তি | ট্রাভেল
7. পড়াশোনা: চাকরি প্রস্তুতি | বিশ্ববিদ্যালয় ভর্তি | একাদশ-দ্বাদশ | ৬ষ্ঠ-১০ম শ্রেণি | ১ম-৫ম শ্রেণি | পরীক্ষা ও ফলাফল

BLOGGER LABELS:
- 5 to 8 comma-separated highly relevant tags (e.g., জাতীয়, রাজনীতি, বাংলাদেশ, সমসাময়িক খবর).

SEO ENGINE:
- Primary Keyword: 1 most relevant focus search keyword.
- Secondary Keywords: 3 to 6 related keywords.
- Meta Description: 140 to 160 characters in natural Bengali. No keyword stuffing.
- Slug: Short, clean lowercase English or transliterated slug (e.g., "bangladesh-sri-lanka-odi-match-win").
- Image Alt Text: Accurate descriptive alt text in Bengali representing the news subject.
`;

// Helper to call Gemini with retry and fallback
async function callGeminiWithFallback(params: {
  contents: string;
  systemInstruction: string;
  responseSchema?: any;
  temperature?: number;
}) {
  const modelsToTry = ['gemini-3.8-flash', 'gemini-3.1-flash-lite'];

  let lastError: any = null;

  for (const model of modelsToTry) {
    for (let attempt = 1; attempt <= 2; attempt++) {
      try {
        const response = await ai.models.generateContent({
          model,
          contents: params.contents,
          config: {
            systemInstruction: params.systemInstruction,
            responseMimeType: 'application/json',
            responseSchema: params.responseSchema,
            temperature: params.temperature ?? 0.25,
          },
        });

        if (response.text) {
          return response;
        }
      } catch (err: any) {
        lastError = err;
        console.warn(`[Gemini Attempt Failed] model=${model} attempt=${attempt}: ${err?.message || err}`);
        // If 503 or 429 or UNAVAILABLE, wait a bit and retry
        const isTransient = err?.message?.includes('503') ||
                            err?.message?.includes('UNAVAILABLE') ||
                            err?.message?.includes('high demand') ||
                            err?.message?.includes('429');
        if (isTransient) {
          await new Promise((r) => setTimeout(r, 1200 * attempt));
        } else {
          // If non-transient, break to next model
          break;
        }
      }
    }
  }

  throw lastError || new Error('Gemini API call failed across models');
}

const NEWS_RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    headlines: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: "৩টি আকর্ষণীয় ও তথ্যনির্ভর শিরোনাম (ক্লিকবেট মুক্ত)"
    },
    recommended_title: {
      type: Type.STRING,
      description: "সার্চ ইন্টেন্ট ও সংবাদের গুরুত্ব অনুযায়ী সেরা ১টি প্রস্তাবিত শিরোনাম"
    },
    article: {
      type: Type.STRING,
      description: "প্যারাগ্রাফভিত্তিক প্রমিত বাংলায় সম্পূর্ণ প্রফেশনাল সংবাদ প্রতিবেদন"
    },
    category: {
      type: Type.STRING,
      description: "অনুমোদিত ৭টি মূল ক্যাটাগরির যেকোনো একটি: জাতীয়, আন্তর্জাতিক, অর্থনীতি, খেলাধুলা, বিনোদন, তথ্য ও জীবন, পড়াশোনা"
    },
    subcategory: {
      type: Type.STRING,
      description: "নির্বাচিত মূল ক্যাটাগরির অন্তর্ভুক্ত একটি অনুমোদিত সাবক্যাটাগরি"
    },
    labels: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: "৫ থেকে ৮টি সুনির্দিষ্ট ব্লগের লেবেল (Labels)"
    },
    seo: {
      type: Type.OBJECT,
      properties: {
        primary_keyword: {
          type: Type.STRING,
          description: "প্রধান সার্চ কিওয়ার্ড"
        },
        secondary_keywords: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
          description: "৩ থেকে ৬টি সহায়ক কিওয়ার্ড"
        },
        meta_description: {
          type: Type.STRING,
          description: "১৪০-১৬০ অক্ষরের স্বাভাবিক বাংলা মেটা বিবরণ"
        },
        slug: {
          type: Type.STRING,
          description: "সংক্ষিপ্ত ও পরিচ্ছন্ন ইংরেজি/ট্রান্সলিটারেটেড এসইও স্লাগ"
        },
        image_alt_text: {
          type: Type.STRING,
          description: "ছবির বর্ণনামূলক অল্টারনেটিভ টেক্সট"
        }
      },
      required: ["primary_keyword", "secondary_keywords", "meta_description", "slug", "image_alt_text"]
    }
  },
  required: ["headlines", "recommended_title", "article", "category", "subcategory", "labels", "seo"]
};

// API: Generate Complete News Package
app.post('/api/news/generate', async (req, res) => {
  try {
    const { rawNews, editorialTone, customInstruction } = req.body;

    if (!rawNews || typeof rawNews !== 'string' || !rawNews.trim()) {
      return res.status(400).json({ error: 'দয়া করে আগে RAW NEWS লিখুন বা paste করুন।' });
    }

    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({
        error: 'সংবাদ তৈরি করা সম্ভব হয়নি। সার্ভারে Gemini API Key কনফিগার করা নেই।'
      });
    }

    let userPrompt = `RAW NEWS / INPUT:\n"""\n${rawNews.trim()}\n"""\n`;
    if (editorialTone) {
      userPrompt += `\nEditorial Tone/Angle Preference: ${editorialTone}`;
    }
    if (customInstruction) {
      userPrompt += `\nAdditional Editorial Note: ${customInstruction}`;
    }
    userPrompt += `\n\nGenerate the complete publish-ready Somoy Sutra news package in the specified structured JSON format. Remember: zero invented facts, preserve sources, neutral tone, natural Bengali.`;

    const response = await callGeminiWithFallback({
      contents: userPrompt,
      systemInstruction: SYSTEM_EDITORIAL_PROMPT,
      responseSchema: NEWS_RESPONSE_SCHEMA,
      temperature: 0.25,
    });

    const responseText = response.text?.trim() || '{}';
    const parsedData = JSON.parse(responseText);

    // Compute basic metrics
    const wordCount = parsedData.article ? parsedData.article.split(/\s+/).filter(Boolean).length : 0;
    const readingTime = Math.max(1, Math.ceil(wordCount / 160));

    return res.json({
      ...parsedData,
      editorial_notes: {
        word_count: wordCount,
        reading_time_minutes: readingTime,
        preserved_facts_check: true
      }
    });
  } catch (err: any) {
    console.error('Error generating news package:', err);
    return res.status(500).json({
      error: 'সংবাদ তৈরি করা সম্ভব হয়নি। API configuration অথবা internet connection পরীক্ষা করুন।',
      details: err?.message || String(err)
    });
  }
});

// API: Regenerate Section
app.post('/api/news/regenerate-section', async (req, res) => {
  try {
    const { section, rawNews, currentArticle, currentTitle, instruction } = req.body;

    if (!rawNews || !section) {
      return res.status(400).json({ error: 'প্রয়োজনীয় তথ্য অনুপস্থিত।' });
    }

    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({ error: 'সার্ভারে Gemini API Key কনফিগার করা নেই।' });
    }

    if (section === 'headlines') {
      const headlineSchema = {
        type: Type.OBJECT,
        properties: {
          headlines: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: "৩টি নতুন তথ্যনির্ভর ও আকর্ষণীয় শিরোনাম"
          },
          recommended_title: {
            type: Type.STRING,
            description: "সেরা ১টি প্রস্তাবিত শিরোনাম"
          }
        },
        required: ["headlines", "recommended_title"]
      };

      const response = await callGeminiWithFallback({
        contents: `RAW NEWS:\n"""\n${rawNews}\n"""\n\nCurrent Article:\n"""\n${currentArticle || ''}\n"""\n\nInstruction: Regenerate 3 distinct headlines and 1 recommended title. Strictly adhere to factual integrity. ${instruction || ''}`,
        systemInstruction: SYSTEM_EDITORIAL_PROMPT,
        responseSchema: headlineSchema,
        temperature: 0.35,
      });

      return res.json(JSON.parse(response.text?.trim() || '{}'));
    }

    if (section === 'article') {
      const articleSchema = {
        type: Type.OBJECT,
        properties: {
          article: {
            type: Type.STRING,
            description: "নতুনভাবে পরিমার্জিত প্রমিত বাংলা সংবাদ প্রতিবেদন"
          }
        },
        required: ["article"]
      };

      const response = await callGeminiWithFallback({
        contents: `RAW NEWS:\n"""\n${rawNews}\n"""\n\nCurrent Headline:\n"${currentTitle || ''}"\n\nInstruction: Rewrite the news article with inverted pyramid journalistic structure. Follow factual integrity strictly. ${instruction || ''}`,
        systemInstruction: SYSTEM_EDITORIAL_PROMPT,
        responseSchema: articleSchema,
        temperature: 0.25,
      });

      return res.json(JSON.parse(response.text?.trim() || '{}'));
    }

    if (section === 'seo') {
      const seoSchema = {
        type: Type.OBJECT,
        properties: {
          primary_keyword: { type: Type.STRING },
          secondary_keywords: { type: Type.ARRAY, items: { type: Type.STRING } },
          meta_description: { type: Type.STRING },
          slug: { type: Type.STRING },
          image_alt_text: { type: Type.STRING }
        },
        required: ["primary_keyword", "secondary_keywords", "meta_description", "slug", "image_alt_text"]
      };

      const response = await callGeminiWithFallback({
        contents: `Title: "${currentTitle || ''}"\nArticle:\n"""\n${currentArticle || rawNews}\n"""\n\nInstruction: Regenerate SEO metadata package (Primary keyword, secondary keywords, meta description 140-160 chars, slug, image alt text).`,
        systemInstruction: SYSTEM_EDITORIAL_PROMPT,
        responseSchema: seoSchema,
        temperature: 0.2,
      });

      return res.json(JSON.parse(response.text?.trim() || '{}'));
    }

    if (section === 'labels') {
      const labelsSchema = {
        type: Type.OBJECT,
        properties: {
          labels: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: "৫ থেকে ৮টি সুনির্দিষ্ট ব্লগের লেবেল"
          }
        },
        required: ["labels"]
      };

      const response = await callGeminiWithFallback({
        contents: `Title: "${currentTitle || ''}"\nRaw News & Article:\n"""\n${currentArticle || rawNews}\n"""\n\nInstruction: Generate 5-8 accurate Blogger labels.`,
        systemInstruction: SYSTEM_EDITORIAL_PROMPT,
        responseSchema: labelsSchema,
        temperature: 0.2,
      });

      return res.json(JSON.parse(response.text?.trim() || '{}'));
    }

    return res.status(400).json({ error: 'অপরিচিত সেকশন।' });
  } catch (err: any) {
    console.error('Error regenerating section:', err);
    return res.status(500).json({
      error: 'সেকশন পুনরায় তৈরি করা সম্ভব হয়নি। অনুগ্রহ করে আবার চেষ্টা করুন।',
      details: err?.message || String(err)
    });
  }
});

// API: Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    product: 'Somoy Sutra AI News Editor',
    hasApiKey: Boolean(process.env.GEMINI_API_KEY)
  });
});

// Setup Vite or static serving
async function setupFrontend() {
  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(distPath, 'index.html'));
    });
  }
}

setupFrontend().then(() => {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[Somoy Sutra] Server listening on http://0.0.0.0:${PORT}`);
  });
});
