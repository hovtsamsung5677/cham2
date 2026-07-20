import 'package:flutter/material.dart';

/// Расширенная таблица цветов (hex -> английское имя) для подбора
/// ближайшего названия при работе с пипеткой. НЕ используется в палитре
/// (color_palette_screen.dart) — только для преобразования захваченного
/// пипеткой RGB в понятное FLUX.2 словесное имя.
///
/// Источник: расширенный набор CSS4/X11 + популярные оттенки.
const List<_NamedColor> kExtendedNamedColors = [
  // ---- Красный ----
  _NamedColor(0xFFFF0000, 'red'),
  _NamedColor(0xFFB22222, 'firebrick'),
  _NamedColor(0xFF8B0000, 'dark red'),
  _NamedColor(0xFF800000, 'maroon'),
  _NamedColor(0xFFDC143C, 'crimson'),
  _NamedColor(0xFFF08080, 'light coral'),
  _NamedColor(0xFFFA8072, 'salmon'),
  _NamedColor(0xFFFF4500, 'orange red'),
  _NamedColor(0xFFCD5C5C, 'indian red'),
  _NamedColor(0xFFE9967A, 'dark salmon'),
  // ---- Оранжевый / персиковый ----
  _NamedColor(0xFFFFA500, 'orange'),
  _NamedColor(0xFFFF8C00, 'dark orange'),
  _NamedColor(0xFFFF7F50, 'coral'),
  _NamedColor(0xFFFFDAB9, 'peach puff'),
  _NamedColor(0xFFFFE4C4, 'bisque'),
  _NamedColor(0xFFD2691E, 'chocolate'),
  _NamedColor(0xFFA0522D, 'sienna'),
  _NamedColor(0xFFDEB887, 'burlywood'),
  _NamedColor(0xFFF4A460, 'sandy brown'),
  _NamedColor(0xFFDA70D6, 'orchid'),
  // ---- Жёлтый / золотой ----
  _NamedColor(0xFFFFFF00, 'yellow'),
  _NamedColor(0xFFFFD700, 'gold'),
  _NamedColor(0xFFDAA520, 'goldenrod'),
  _NamedColor(0xFFB8860B, 'dark goldenrod'),
  _NamedColor(0xFFF0E68C, 'khaki'),
  _NamedColor(0xFFEEE8AA, 'pale goldenrod'),
  _NamedColor(0xFFFFFACD, 'lemon chiffon'),
  _NamedColor(0xFFFFF8DC, 'cornsilk'),
  _NamedColor(0xFFF5DEB3, 'wheat'),
  _NamedColor(0xFFFFEF96, 'papaya whip'),
  // ---- Зелёный ----
  _NamedColor(0xFF008000, 'green'),
  _NamedColor(0xFF006400, 'dark green'),
  _NamedColor(0xFF228B22, 'forest green'),
  _NamedColor(0xFF2E8B57, 'sea green'),
  _NamedColor(0xFF3CB371, 'medium sea green'),
  _NamedColor(0xFF00FF00, 'lime'),
  _NamedColor(0xFF32CD32, 'lime green'),
  _NamedColor(0xFF90EE90, 'light green'),
  _NamedColor(0xFF9ACD32, 'yellow green'),
  _NamedColor(0xFF6B8E23, 'olive green'),
  _NamedColor(0xFF808000, 'olive'),
  _NamedColor(0xFF556B2F, 'dark olive green'),
  _NamedColor(0xFF7CFC00, 'lawn green'),
  _NamedColor(0xFFADFF2F, 'green yellow'),
  _NamedColor(0xFF00FA9A, 'medium spring green'),
  _NamedColor(0xFF66CDAA, 'medium aquamarine'),
  // ---- Бирюзовый / циан ----
  _NamedColor(0xFF00FFFF, 'cyan'),
  _NamedColor(0xFFE0FFFF, 'light cyan'),
  _NamedColor(0xFF40E0D0, 'turquoise'),
  _NamedColor(0xFF48D1CC, 'medium turquoise'),
  _NamedColor(0xFF20B2AA, 'light sea green'),
  _NamedColor(0xFF008B8B, 'dark cyan'),
  _NamedColor(0xFF008080, 'teal'),
  _NamedColor(0xFF5F9EA0, 'cadet blue'),
  _NamedColor(0xFF7FFFD4, 'aquamarine'),
  _NamedColor(0xFFAFEEEE, 'pale turquoise'),
  // ---- Синий ----
  _NamedColor(0xFF0000FF, 'blue'),
  _NamedColor(0xFF00008B, 'dark blue'),
  _NamedColor(0xFF000080, 'navy'),
  _NamedColor(0xFF191970, 'midnight blue'),
  _NamedColor(0xFF4169E1, 'royal blue'),
  _NamedColor(0xFF1E90FF, 'dodger blue'),
  _NamedColor(0xFF6495ED, 'cornflower blue'),
  _NamedColor(0xFF87CEEB, 'sky blue'),
  _NamedColor(0xFF87CEFA, 'light sky blue'),
  _NamedColor(0xFFB0C4DE, 'light steel blue'),
  _NamedColor(0xFF4682B4, 'steel blue'),
  _NamedColor(0xFF70A1FF, 'azure'),
  _NamedColor(0xFF00BFFF, 'deep sky blue'),
  _NamedColor(0xFF1E6FFF, 'azure blue'),
  // ---- Фиолетовый / пурпурный ----
  _NamedColor(0xFF800080, 'purple'),
  _NamedColor(0xFF4B0082, 'indigo'),
  _NamedColor(0xFF483D8B, 'dark slate blue'),
  _NamedColor(0xFF6A5ACD, 'slate blue'),
  _NamedColor(0xFF7B68EE, 'medium slate blue'),
  _NamedColor(0xFF8A2BE2, 'blue violet'),
  _NamedColor(0xFF9400D3, 'dark violet'),
  _NamedColor(0xFF9932CC, 'dark orchid'),
  _NamedColor(0xFFBA55D3, 'medium orchid'),
  _NamedColor(0xFFDA70D6, 'orchid'),
  _NamedColor(0xFFEE82EE, 'violet'),
  _NamedColor(0xFFD8BFD8, 'thistle'),
  _NamedColor(0xFFE6E6FA, 'lavender'),
  _NamedColor(0xFFC8A2C8, 'lilac'),
  // ---- Розовый / маджента ----
  _NamedColor(0xFFFF00FF, 'magenta'),
  _NamedColor(0xFFFF1493, 'deep pink'),
  _NamedColor(0xFFFF69B4, 'hot pink'),
  _NamedColor(0xFFFFC0CB, 'pink'),
  _NamedColor(0xFFFFB6C1, 'light pink'),
  _NamedColor(0xFFDB7093, 'pale violet red'),
  _NamedColor(0xFFC71585, 'medium violet red'),
  _NamedColor(0xFFFFC0CB, 'rose'),
  _NamedColor(0xFFF4C2C2, 'pinkish'),
  // ---- Коричневый / земляной ----
  _NamedColor(0xFFA52A2A, 'brown'),
  _NamedColor(0xFF8B4513, 'saddle brown'),
  _NamedColor(0xFFA0522D, 'sienna'),
  _NamedColor(0xFFD2691E, 'bronze'),
  _NamedColor(0xFFCD853F, 'peru'),
  _NamedColor(0xFFB5651D, 'dark bronze'),
  _NamedColor(0xFFEED6AF, 'blanched almond'),
  _NamedColor(0xFFC19A6B, 'camel'),
  _NamedColor(0xFF7B3F00, 'dark brown'),
  _NamedColor(0xFF5C3A21, 'espresso'),
  _NamedColor(0xFF9B7653, 'taupe'),
  _NamedColor(0xFF704214, 'dark oak'),
  // ---- Серый / металлы ----
  _NamedColor(0xFF808080, 'gray'),
  _NamedColor(0xFFA9A9A9, 'dark gray'),
  _NamedColor(0xFFD3D3D3, 'light gray'),
  _NamedColor(0xFF696969, 'dim gray'),
  _NamedColor(0xFFC0C0C0, 'silver'),
  _NamedColor(0xFFE0E0E0, 'light silver'),
  _NamedColor(0xFFF5F5F5, 'white smoke'),
  _NamedColor(0xFFFAFAFA, 'off white'),
  _NamedColor(0xFFFFFFFF, 'white'),
  _NamedColor(0xFF000000, 'black'),
  _NamedColor(0xFF1A1A1A, 'rich black'),
  _NamedColor(0xFF2F2F2F, 'charcoal'),
  _NamedColor(0xFF3A3A3A, 'dark charcoal'),
  _NamedColor(0xFF464646, 'graphite'),
  _NamedColor(0xFF6E7478, 'titanium'),
  _NamedColor(0xFFB0BEC5, 'blue gray'),
  _NamedColor(0xFF78909C, 'blue grey'),
  _NamedColor(0xFF455A64, 'blueish gray'),
  _NamedColor(0xFFCD7F32, 'copper'),
  _NamedColor(0xFFC9A66B, 'brass'),
  _NamedColor(0xFFE8ECEF, 'stainless steel'),
  _NamedColor(0xFFD4AF37, 'metallic gold'),
  _NamedColor(0xFFB87333, 'metallic copper'),
];

class _NamedColor {
  final int hex;
  final String name;
  const _NamedColor(this.hex, this.name);
}

/// Возвращает ближайшее словесное имя цвета (по евклидову расстоянию в RGB)
/// для захваченного пипеткой цвета. Если цвет почти серый — подбирает
/// соответствующий серый/металлический оттенок, иначе — ближайший цветной.
String nearestColorName(Color color) {
  final int r = color.red;
  final int g = color.green;
  final int b = color.blue;

  // Всегда ищем ближайший цвет по евклидову расстоянию в RGB по ВСЕЙ
  // таблице (в ней уже есть серые, белые, чёрные и металлические оттенки).
  // Отдельную «серую» ветку не делаем: бледно-розовый/бледно-голубой при
  // низкой насыщенности иначе ошибочно попадал в «gray».
  int bestDist = 1 << 30;
  String bestName = kExtendedNamedColors.first.name;
  for (final named in kExtendedNamedColors) {
    final int cr = (named.hex >> 16) & 0xFF;
    final int cg = (named.hex >> 8) & 0xFF;
    final int cb = named.hex & 0xFF;
    final int dr = r - cr;
    final int dg = g - cg;
    final int db = b - cb;
    final int dist = dr * dr + dg * dg + db * db;
    if (dist < bestDist) {
      bestDist = dist;
      bestName = named.name;
    }
  }
  return bestName;
}
