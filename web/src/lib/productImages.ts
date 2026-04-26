/** Stable Unsplash URLs by grocery category (for product cards). */

const DEFAULT =
  "https://images.unsplash.com/photo-1604719314766-9042a01e64ce?w=800&h=500&fit=crop&q=80"

const BY_CATEGORY: Record<string, string> = {
  PRODUCE:
    "https://images.unsplash.com/photo-1542838132-92c53300491e?w=800&h=500&fit=crop&q=80",
  FRUITS:
    "https://images.unsplash.com/photo-1619566636858-adf3ef46400b?w=800&h=500&fit=crop&q=80",
  VEGETABLES:
    "https://images.unsplash.com/photo-1597362925123-77861d3b1c66?w=800&h=500&fit=crop&q=80",
  DAIRY:
    "https://images.unsplash.com/photo-1563636619-e9143da7973b?w=800&h=500&fit=crop&q=80",
  BAKERY:
    "https://images.unsplash.com/photo-1509440159596-0249088772ff?w=800&h=500&fit=crop&q=80",
  MEAT:
    "https://images.unsplash.com/photo-1603048297172-c92544798d5a?w=800&h=500&fit=crop&q=80",
  SEAFOOD:
    "https://images.unsplash.com/photo-1559339352-11d035aa65de?w=800&h=500&fit=crop&q=80",
  BEVERAGES:
    "https://images.unsplash.com/photo-1629203851122-3726ecdf080e?w=800&h=500&fit=crop&q=80",
  SNACKS:
    "https://images.unsplash.com/photo-1625246333195-78d9c38ad449?w=800&h=500&fit=crop&q=80",
  FROZEN:
    "https://images.unsplash.com/photo-1574943320859-4d32d6a9575a?w=800&h=500&fit=crop&q=80",
  PACKAGED:
    "https://images.unsplash.com/photo-1586201375761-83865001e31c?w=800&h=500&fit=crop&q=80",
  GROCERY:
    "https://images.unsplash.com/photo-1588964895597-cfccd6e2dbf9?w=800&h=500&fit=crop&q=80",
}

export function imageForCategory(category: string): string {
  const key = category.toUpperCase().replace(/[^A-Z]/g, "")
  for (const [k, url] of Object.entries(BY_CATEGORY)) {
    if (key.includes(k) || k.includes(key)) return url
  }
  return DEFAULT
}
