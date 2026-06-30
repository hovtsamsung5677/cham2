import 'package:flutter/material.dart';

class ColorPaletteScreen extends StatefulWidget {
  const ColorPaletteScreen({super.key});

  @override
  State<ColorPaletteScreen> createState() => _ColorPaletteScreenState();
}

class _ColorPaletteScreenState extends State<ColorPaletteScreen> {
  int? selectedPaletteIndex;

  List<Color> get _currentColors {
    return [
      ...solidColors,
      ...plasticColors,
      ...fabricColors,
      ...woodColors,
      ...metalTintColors,
      ...glassColors,
      ...leatherColors,
      ...ceramicColors,
      ...concreteColors,
    ];
  }

  List<Color> get _sortedByHue {
    final colors = List<Color>.from(_currentColors);
    colors.sort((a, b) {
      final hslA = _colorToHSL(a);
      final hslB = _colorToHSL(b);
      if ((hslA['h']! - hslB['h']!).abs() < 15) {
        return (hslA['l']!).compareTo(hslB['l']!);
      }
      return hslA['h']!.compareTo(hslB['h']!);
    });
    return colors;
  }

  Map<String, double> _colorToHSL(Color color) {
    final r = color.red / 255.0;
    final g = color.green / 255.0;
    final b = color.blue / 255.0;
    final max = r > g ? (r > b ? r : b) : (g > b ? g : b);
    final min = r < g ? (r < b ? r : b) : (g < b ? g : b);
    final l = (max + min) / 2.0;

    if (max == min) {
      return {'h': 0.0, 's': 0.0, 'l': l};
    }

    final d = max - min;
    final s = l > 0.5 ? d / (2.0 - max - min) : d / (max + min);
    double h;
    if (max == r) {
      h = (g - b) / d + (g < b ? 6 : 0);
    } else if (max == g) {
      h = (b - r) / d + 2.0;
    } else {
      h = (r - g) / d + 4.0;
    }
    h = (h / 6.0) * 360.0;

    return {'h': h, 's': s, 'l': l};
  }

  final List<Color> solidColors = [
    const Color(0xFF8B4513),
    const Color(0xFFE040FB),
    const Color(0xFF2196F3),
    const Color(0xFF00BCD4),
    const Color(0xFF4CAF50),
    const Color(0xFFF44336),
    const Color(0xFF9C27B0),
    const Color(0xFFCDDC39),
    const Color(0xFFFFEB3B),
    const Color(0xFFFF9800),
    const Color(0xFF795548),
    const Color(0xFF607D8B),
  ];

  final List<Color> plasticColors = [
    const Color(0xFFE53935),
    const Color(0xFFFB8C00),
    const Color(0xFFFDD835),
    const Color(0xFF43A047),
    const Color(0xFF00ACC1),
    const Color(0xFF1E88E5),
    const Color(0xFF5E35B1),
    const Color(0xFF8E24AA),
    const Color(0xFFD81B60),
    const Color(0xFF6D4C41),
    const Color(0xFFF5F5F5),
    const Color(0xFF212121),
  ];

  final List<Color> fabricColors = [
    const Color(0xFF8D6E63),
    const Color(0xFFA1887F),
    const Color(0xFFBCAAA4),
    const Color(0xFFD7CCC8),
    const Color(0xFF37474F),
    const Color(0xFF546E7A),
    const Color(0xFF455A64),
    const Color(0xFF607D8B),
    const Color(0xFF795548),
    const Color(0xFF4E342E),
    const Color(0xFF263238),
    const Color(0xFFECEFF1),
  ];

  final List<Color> woodColors = [
    const Color(0xFF8B4513),
    const Color(0xFFA0522D),
    const Color(0xFFD2691E),
    const Color(0xFFBC8F8F),
    const Color(0xFFCD853F),
    const Color(0xFFDEB887),
    const Color(0xFFF4A460),
    const Color(0xFFD2B48C),
  ];

  final List<Color> metalTintColors = [
    const Color(0xFFEBB014),
    const Color(0xFFC0C0C0),
    const Color(0xFF984D25),
  ];

  final List<Color> glassColors = [
    const Color(0xFFE1F5FE),
    const Color(0xFFF3E5F5),
    const Color(0xFFE8F5E9),
    const Color(0xFFFFF8E1),
    const Color(0xFFFFEBEE),
    const Color(0xFFF5F5F5),
    const Color(0xFFB3E5FC),
    const Color(0xFFCE93D8),
  ];

  final List<Color> leatherColors = [
    const Color(0xFF8B4513),
    const Color(0xFFA0522D),
    const Color(0xFFCD853F),
    const Color(0xFFDEB887),
    const Color(0xFFD2691E),
    const Color(0xFFF5F5DC),
    const Color(0xFF2F1B14),
    const Color(0xFF4A2C2A),
  ];

  final List<Color> ceramicColors = [
    const Color(0xFFFFFFFF),
    const Color(0xFFF5F5F5),
    const Color(0xFFE8E8E8),
    const Color(0xFFB0BEC5),
    const Color(0xFF90A4AE),
    const Color(0xFF607D8B),
    const Color(0xFF795548),
    const Color(0xFFFDD835),
  ];

  final List<Color> concreteColors = [
    const Color(0xFF9E9E9E),
    const Color(0xFFBDBDBD),
    const Color(0xFF757575),
    const Color(0xFF616161),
    const Color(0xFFE0E0E0),
    const Color(0xFF424242),
    const Color(0xFFF5F5F5),
    const Color(0xFF795548),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1C1C1E),
      body: SafeArea(
        child: Column(
          children: [
            _buildTopBar(),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 8),
                    _buildColorTextureSection(),
                    const SizedBox(height: 24),
                  ],
                ),
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
            child: const Icon(Icons.arrow_back, color: Colors.white, size: 24),
          ),
          Image.asset(
            'assets/icons/Squared_Menu.png',
            width: 28,
            height: 28,
            color: Colors.white,
            errorBuilder: (context, error, stackTrace) =>
                const Icon(Icons.grid_view, color: Colors.white, size: 28),
          ),
          GestureDetector(
            onTap: selectedPaletteIndex != null
                ? () {
                    final color = _sortedByHue[selectedPaletteIndex!];
                    Navigator.pop(context, color);
                  }
                : null,
            child: Icon(
              Icons.check,
              color: selectedPaletteIndex != null
                  ? Colors.white
                  : Colors.white38,
              size: 24,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildColorTextureSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Icon(Icons.palette_outlined, color: Colors.white, size: 22),
            const SizedBox(width: 8),
            const Text(
              'Палитра цветов',
              style: TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
        const SizedBox(height: 14),
        _buildColorGrid(),
      ],
    );
  }

  Widget _buildColorGrid() {
    final colors = _sortedByHue;
    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 4,
        crossAxisSpacing: 8,
        mainAxisSpacing: 8,
        childAspectRatio: 1.4,
      ),
      itemCount: colors.length,
      itemBuilder: (context, index) {
        final isSelected = selectedPaletteIndex == index;
        final color = colors[index];
        return GestureDetector(
          onTap: () {
            setState(() {
              selectedPaletteIndex = index;
            });
          },
          child: _buildColorTile(color, isSelected),
        );
      },
    );
  }

  Widget _buildColorTile(Color color, bool isSelected) {
    return Stack(
      children: [
        Container(
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        if (isSelected)
          Positioned.fill(
            child: Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.white, width: 2.5),
              ),
              child: const Center(
                child: Icon(Icons.check, color: Colors.white, size: 20),
              ),
            ),
          ),
      ],
    );
  }
}
