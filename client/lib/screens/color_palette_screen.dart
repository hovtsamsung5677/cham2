import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/app_state.dart';

class ColorPaletteScreen extends StatefulWidget {
  const ColorPaletteScreen({super.key});

  @override
  State<ColorPaletteScreen> createState() => _ColorPaletteScreenState();
}

class _ColorPaletteScreenState extends State<ColorPaletteScreen> {
  int? _expandedIndex = 0;
  Color? _selectedColor;

  static const _bg = Color(0xFF151412);
  static const _tileColor = Color(0xFF1E1E1E);

  final List<_ColorCategory> _colorCategories = [
    _ColorCategory('Красный', [
      const Color(0xFFE53935),
      const Color(0xFFB71C1C),
      const Color(0xFFEF9A9A),
      const Color(0xFFFF5252),
      const Color(0xFFD32F2F),
    ]),
    _ColorCategory('Оранжевый', [
      const Color(0xFFFF6D00),
      const Color(0xFFB33E12),
      const Color(0xFFFFC38A),
      const Color(0xFFF08A1E),
      const Color(0xFFFF9800),
    ]),
    _ColorCategory('Желтый', [
      const Color(0xFFFFEB3B),
      const Color(0xFFFBC02D),
      const Color(0xFFFFF59D),
      const Color(0xFFF9A825),
      const Color(0xFFFFD600),
    ]),
    _ColorCategory('Зеленый', [
      const Color(0xFF4CAF50),
      const Color(0xFF1B5E20),
      const Color(0xFFA5D6A7),
      const Color(0xFF66BB6A),
      const Color(0xFF2E7D32),
    ]),
    _ColorCategory('Голубой', [
      const Color(0xFF29B6F6),
      const Color(0xFF01579B),
      const Color(0xFFB3E5FC),
      const Color(0xFF03A9F4),
      const Color(0xFF0288D1),
    ]),
    _ColorCategory('Синий', [
      const Color(0xFF3F51B5),
      const Color(0xFF1A237E),
      const Color(0xFF9FA8DA),
      const Color(0xFF3949AB),
      const Color(0xFF283593),
    ]),
    _ColorCategory('Фиолетовый', [
      const Color(0xFF9C27B0),
      const Color(0xFF4A148C),
      const Color(0xFFCE93D8),
      const Color(0xFFAB47BC),
      const Color(0xFF7B1FA2),
    ]),
    _ColorCategory(
      'Металл',
      [
        const Color(0xFFD4AF37), // Золото
        const Color(0xFFC0C0C0), // Серебро
        const Color(0xFFCD7F32), // Бронза
        const Color(0xFFB5A642), // Латунь
        const Color(0xFF878681), // Титан
        const Color(0xFFCED4D8), // Нержавейка
        const Color(0xFFB87333), // Медь
      ],
      labels: const [
        'Золото',
        'Серебро',
        'Бронза',
        'Латунь',
        'Титан',
        'Нержавейка',
        'Медь',
      ],
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: _bg,
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      child: SafeArea(
        top: false,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(height: 8),
            Container(
              width: 36,
              height: 4,
              decoration: BoxDecoration(
                color: Colors.white24,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: 16),
            _buildTopBar(),
            Flexible(
              child: SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: _buildCategoriesList(),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTopBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          GestureDetector(
            onTap: () => Navigator.pop(context),
            child: Container(
              width: 44,
              height: 44,
              decoration: const BoxDecoration(
                color: Colors.white12,
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.arrow_back, color: Colors.white, size: 22),
            ),
          ),
          const _PaletteIconInFrame(),
          GestureDetector(
            onTap: _selectedColor != null
                ? () => Navigator.pop(context, _selectedColor)
                : null,
            child: Icon(
              Icons.check,
              color: _selectedColor != null ? Colors.white : Colors.white38,
              size: 24,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCategoriesList() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 8),
        ...List.generate(_colorCategories.length, (index) {
          final category = _colorCategories[index];
          final isExpanded = _expandedIndex == index;
          final isMetal = category.name == 'Металл';
          return Container(
            margin: const EdgeInsets.only(bottom: 12),
            decoration: BoxDecoration(
              color: _tileColor,
              borderRadius: BorderRadius.circular(16),
              border: isExpanded
                  ? Border.all(color: Colors.white, width: 1.2)
                  : null,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: InkWell(
                        borderRadius: BorderRadius.circular(16),
                        onTap: () {
                          setState(() {
                            _expandedIndex = isExpanded ? null : index;
                          });
                        },
                        child: Padding(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 20,
                            vertical: 18,
                          ),
                          child: Row(
                            children: [
                              Expanded(
                                child: Text(
                                  category.name,
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 22,
                                  ),
                                ),
                              ),
                              Icon(
                                isExpanded
                                    ? Icons.keyboard_arrow_up
                                    : Icons.keyboard_arrow_down,
                                color: Colors.white,
                                size: 26,
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                    if (isMetal) _buildPatinaToggle(),
                  ],
                ),
                AnimatedCrossFade(
                  duration: const Duration(milliseconds: 200),
                  crossFadeState: isExpanded
                      ? CrossFadeState.showFirst
                      : CrossFadeState.showSecond,
                  firstChild: Padding(
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                    child: _ColorGrid(
                      colors: category.shades,
                      labels: category.labels,
                      selectedColor: _selectedColor,
                      categoryName: category.name,
                      onColorTap: (color) {
                        setState(() {
                          _selectedColor = color;
                        });
                      },
                    ),
                  ),
                  secondChild: const SizedBox(width: double.infinity),
                ),
              ],
            ),
          );
        }),
      ],
    );
  }

  Widget _buildPatinaToggle() {
    return Consumer<AppState>(
      builder: (context, appState, _) {
        return Padding(
          padding: const EdgeInsets.only(right: 16),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                'Патина',
                style: TextStyle(color: Colors.white70, fontSize: 13),
              ),
              Switch(
                value: appState.patinaMode,
                materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                onChanged: (value) {
                  appState.setPatinaMode(value);
                },
              ),
            ],
          ),
        );
      },
    );
  }
}

class _ColorGrid extends StatelessWidget {
  final List<Color> colors;
  final Color? selectedColor;
  final ValueChanged<Color> onColorTap;
  final String? categoryName;
  final List<String>? labels;

  const _ColorGrid({
    required this.colors,
    required this.selectedColor,
    required this.onColorTap,
    this.categoryName,
    this.labels,
  });

  @override
  Widget build(BuildContext context) {
    final isMetal = categoryName == 'Металл';
    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: colors.length,
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
        childAspectRatio: 1.9,
      ),
      itemBuilder: (context, index) {
        final color = colors[index];
        final isSelected = selectedColor == color;
        final label = (labels != null && index < labels!.length)
            ? labels![index]
            : null;

        return GestureDetector(
          onTap: () => onColorTap(color),
          child: Container(
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(10),
              border: isSelected
                  ? Border.all(color: Colors.white, width: 2.5)
                  : null,
            ),
            clipBehavior: Clip.antiAlias,
            child: Stack(
              fit: StackFit.expand,
              children: [
                // Металлический рельеф поверх реального цвета (цвет не искажается)
                if (isMetal)
                  Opacity(
                    opacity: 0.18,
                    child: Image.asset(
                      'assets/textures/metal_texture.png',
                      fit: BoxFit.cover,
                    ),
                  ),
                // Название плитки белым цветом (только у металлов)
                if (label != null)
                  Center(
                    child: Text(
                      label,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                        shadows: [
                          Shadow(
                            color: Colors.black87,
                            blurRadius: 4,
                            offset: Offset(0, 1),
                          ),
                        ],
                      ),
                    ),
                  ),
                // Отметка выбранного
                if (isSelected)
                  const Center(
                    child: Icon(Icons.check, color: Colors.white),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _ColorCategory {
  final String name;
  final List<Color> shades;
  final List<String>? labels;

  _ColorCategory(this.name, this.shades, {this.labels});
}

class _PaletteIconInFrame extends StatelessWidget {
  const _PaletteIconInFrame();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 28,
      height: 28,
      decoration: const BoxDecoration(
        image: DecorationImage(
          image: AssetImage('assets/icons/ramka.png'),
          fit: BoxFit.cover,
        ),
      ),
      child: Center(
        child: Image.asset(
          'assets/icons/Paint Palette.png',
          width: 18,
          height: 18,
          color: Colors.white,
        ),
      ),
    );
  }
}
