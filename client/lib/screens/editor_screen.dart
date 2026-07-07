import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/app_state.dart';
import '../models/selection_tool.dart';
import '../widgets/selection_canvas.dart';
import '../services/segmentation_service.dart';
import 'color_picker_screen.dart';
import 'color_palette_screen.dart';
import 'material_selection_screen.dart';
import '../utils/transitions.dart';
import 'camera_page.dart';
import 'export_screen.dart';
import 'projects_screen.dart';

class EditorScreen extends StatefulWidget {
  const EditorScreen({super.key});

  @override
  State<EditorScreen> createState() => _EditorScreenState();
}

class _EditorScreenState extends State<EditorScreen>
    with TickerProviderStateMixin {
  // Local state
  SelectionTool _selectedTool = SelectionTool.interactiveSegmentation;
  final double _brushSize = 30;

  // Segmentation service
  late final SegmentationService _segmentationService;

  // FAB animation
  late AnimationController _fabPulseController;
  late Animation<double> _fabPulseAnimation;

  // State for segmentation mode (toggle)
  bool _isSegmentationModeActive = false;

  // State for recolor quality
  bool _isComplexRecolorMode = false;

  // FAB initialization (first press just activates)
  bool _fabInitialized = false;

  // Guard against multiple concurrent segmentations
  bool _isProcessing = false;

  // Last tap position for AI recolor fallback
  Offset? _lastTapImagePosition;
  Size? _lastImageSize;
  Uint8List? _lastImageBytes;

  @override
  void initState() {
    super.initState();
    _segmentationService = SegmentationService();
    _fabPulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat(reverse: true);
    _fabPulseAnimation = Tween<double>(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(parent: _fabPulseController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _segmentationService.dispose();
    _fabPulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF2C2C2E),
      body: Stack(
        children: [
          // Canvas area — isolated rebuild scope via RepaintBoundary
          Positioned.fill(
            bottom: 220,
             child: Consumer<AppState>(
              builder: (context, appState, child) {
                final imageBytes = appState.capturedImage;
                final previewBytes = appState.previewImage;

                if (imageBytes == null) {
                  return const _EmptyCanvasPlaceholder();
                }

                return RepaintBoundary(
                  child: MouseRegion(
                    cursor: SystemMouseCursors.basic,
                    child: SelectionCanvas(
                      key: const ValueKey('selection_canvas'),
                      imageBytes: imageBytes,
                      previewImage: previewBytes,
                      selectionMask: (appState.isPreviewMode && appState.previewImage != null) ? Uint8List(0) : appState.selectionMask,
                      currentTool: _selectedTool,
                      brushSize: _brushSize,
                      lassoPoints: const [],
                      polygonPoints: const [],
                      rectanglePoints: const [],
                      boundaryPoints: const [],
                      onSelectionUpdate: appState.setSelectionMask,
                      onLassoPointsUpdate: (_) {},
                      onPolygonPointsUpdate: (_) {},
                      onRectanglePointsUpdate: (_) {},
                      onBoundaryStart: null,
                      onBoundaryPoint: null,
                      onBoundaryEnd: null,
                      onDrawingStart: () {},
                      onDrawingEnd: () {},
                      onAutoSegmentTap:
                          _selectedTool ==
                                  SelectionTool.interactiveSegmentation &&
                              _fabInitialized &&
                              _isSegmentationModeActive
                          ? _handleAutoSegmentation
                          : null,
                      isSegmentationModeActive: _isSegmentationModeActive,
                    ),
                  ),
                );
              },
            ),
          ),

// Top toolbar — extracted to own widget to avoid canvas rebuilds
          _EditorTopToolbar(
            onBackToCamera: () => _onBackToCamera(context),
            onGoHome: () => Navigator.push(
              context,
              AppTransitions.fadeRoute(const ProjectsScreen()),
            ),
          ),

          // Bottom panel — state method for direct access to stateful FAB
          _buildBottomPanel(),

          // Loading overlay
          Consumer<AppState>(
            builder: (context, appState, child) {
              if (!appState.isLoading) return const SizedBox.shrink();
              return Container(
                color: Colors.black54,
                child: Center(
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1C1C1E),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: const Color(0xFFFFC107).withOpacity(0.25)),
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: const [
                        CircularProgressIndicator(color: Color(0xFFF5C518)),
                        SizedBox(height: 14),
                        Text(
                          'AI перекраска...',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // BOTTOM PANEL
  // ═══════════════════════════════════════════════════════════════════════════

  Widget _buildBottomPanel() {
    return Align(
      alignment: Alignment.bottomCenter,
      child: Container(
        width: double.infinity,
        decoration: const BoxDecoration(
          color: Color(0xFF1C1C1E),
          borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
        ),
        padding: const EdgeInsets.fromLTRB(0, 12, 0, 32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Drag handle
            Container(
              width: 36,
              height: 4,
              decoration: BoxDecoration(
                color: Colors.white24,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
             const SizedBox(height: 14),
              SizedBox(
                height: 68,
                child: Stack(
                  children: [
                    const Center(child: SizedBox(width: 68, height: 68)),
                    Center(child: _buildAutoSegmentationFAB()),
                    Align(
                      alignment: Alignment.centerRight,
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            _isComplexRecolorMode ? 'Сложная' : 'Простая',
                            style: const TextStyle(color: Colors.white70, fontSize: 10),
                          ),
                           Switch(
                             value: _isComplexRecolorMode,
                             materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                             onChanged: (val) {
                               setState(() {
                                 _isComplexRecolorMode = val;
                               });
                             },
                           ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
               const SizedBox(height: 36),
              // Bottom actions row - evenly spaced, middle item under FAB
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _BottomAction(
                    child: const _ColorPreviewWidget(),
                    label: 'Цвет',
                    onTap: () => _showColorPicker(context),
                  ),
                  _BottomAction(
                   child: const _IconInFrameWidget(
                     assetPath: 'assets/icons/Paint Palette.png',
                     size: 48,
                   ),
                   label: 'Палитра',
                   onTap: () => _showColorPalette(context),
                 ),
                 _BottomAction(
                   child: const _IconInFrameWidget(
                     assetPath: 'assets/icons/Diagonal Lines.png',
                     size: 48,
                   ),
                   label: 'Материал',
                   onTap: () => _showMaterialSelection(context),
                 ),
               ],
             ),
          ],
        ),
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // AUTO-SEGMENTATION FAB
  // ═══════════════════════════════════════════════════════════════════════════

  Widget _buildAutoSegmentationFAB() {
    final bool isActive = _isSegmentationModeActive && _fabInitialized;

    return AnimatedBuilder(
      animation: _fabPulseAnimation,
      builder: (context, child) {
        return Transform.scale(
          scale: _fabPulseAnimation.value,
          child: child,
        );
      },
child: GestureDetector(
         onTap: () {
           if (_isProcessing) return;
           setState(() {
             if (!_fabInitialized) {
               _fabInitialized = true;
               _isSegmentationModeActive = true;
             } else {
               _isSegmentationModeActive = !_isSegmentationModeActive;
             }
           });
         },
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOutBack,
          width: 68,
          height: 68,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: isActive ? const Color(0xFFFFC107) : Colors.grey,
            boxShadow: isActive
                ? [
                    BoxShadow(
                      color: const Color(0xFFFFC107).withValues(alpha: 0.5),
                      blurRadius: 20,
                      spreadRadius: 4,
                    ),
                  ]
                : [],
          ),
          child: TweenAnimationBuilder<double>(
            duration: const Duration(milliseconds: 200),
            tween: Tween(begin: 0.0, end: 1.0),
            curve: Curves.easeOut,
            builder: (context, value, child) {
              return Opacity(opacity: value, child: child);
            },
            child: Center(
              child: Image.asset(
                'assets/icons/Hand Cursor.png',
                width: 32,
                height: 32,
                color: isActive ? Colors.white : Colors.white70,
              ),
            ),
          ),
        ),
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SEGMENTATION
  // ═══════════════════════════════════════════════════════════════════════════

/// Обрабатывает клик для авто-сегментации объекта с AI-перекраской.
  /// Координаты уже преобразованы в пространство исходного изображения.
  Future<void> _handleAutoSegmentation(Uint8List orientedBytes, Offset imagePosition, int imageWidth, int imageHeight) async {
    _lastTapImagePosition = imagePosition;
    _lastImageSize = Size(imageWidth.toDouble(), imageHeight.toDouble());
    _lastImageBytes = orientedBytes;
    await _runAIRecolor(orientedBytes, imagePosition, Size(imageWidth.toDouble(), imageHeight.toDouble()));
  }

Future<void> _runAIRecolor(Uint8List orientedBytes, Offset imagePosition, Size imageSize) async {
    if (_isProcessing) return;
    _isProcessing = true;

    final appState = context.read<AppState>();

    if (appState.isLoading) {
      _isProcessing = false;
      return;
    }

    appState.setLoading(true);
    appState.setPreviewImage(null);
    if (appState.isPreviewMode) appState.togglePreviewMode();

    try {
      // Use already-oriented bytes from the canvas
      debugPrint('AI recolor: position=$imagePosition, imageSize=$imageSize');

      final resultBytes = await _segmentationService.segmentObject(
          imageBytes: orientedBytes,
          imagePosition: imagePosition,
          imageWidth: imageSize.width.toInt(),
          imageHeight: imageSize.height.toInt(),
          material: appState.selectedMaterial,
          colorHex: appState.selectedColor.value,
          objectName: 'object',
          strength: 1.0,
          guidanceScale: 5.0,
          numInferenceSteps: _isComplexRecolorMode ? 30 : 6,
        );

      if (!mounted) {
        _isProcessing = false;
        appState.setLoading(false);
        return;
      }

      if (resultBytes != null) {
        appState.setPreviewImage(resultBytes);
        if (!appState.isPreviewMode) appState.togglePreviewMode();
        appState.addProject(resultBytes);

        final imageProvider = MemoryImage(resultBytes);
        await precacheImage(imageProvider, context);

        if (mounted) {
          Navigator.push(
            context,
            AppTransitions.slideRoute(
              const ExportScreen(),
              direction: SlideDirection.up,
            ),
          );
        }
      } else {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('Ошибка AI перекраски')));
      }
    } catch (e) {
      debugPrint('Ошибка AI: $e');
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Ошибка AI: $e')));
      }
    } finally {
      _isProcessing = false;
      if (mounted) {
        appState.setLoading(false);
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // NAVIGATION / ACTION HELPERS
  // ═══════════════════════════════════════════════════════════════════════════

  void _showMaterialSelection(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => const MaterialSelectionScreen(),
    );
  }

  void _showColorPicker(BuildContext context) async {
    final appState = context.read<AppState>();
    final result = await showModalBottomSheet<Color?>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => ColorPickerScreen(
        initialColor: appState.selectedColor,
        onColorChanged: (color) {
          appState.setSelectedColor(color);
        },
      ),
    );
    if (!mounted) return;
    if (result != null) {
      appState.setSelectedColor(result);
    }
  }

  Future<void> _showColorPalette(BuildContext context) async {
    final appState = context.read<AppState>();
    final result = await showModalBottomSheet<Color?>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => const ColorPaletteScreen(),
    );
    if (!mounted) return;
    if (result != null) {
      appState.setSelectedColor(result);
      if (_lastImageBytes != null && _lastTapImagePosition != null && _lastImageSize != null) {
        await _runAIRecolor(_lastImageBytes!, _lastTapImagePosition!, _lastImageSize!);
      }
    }
  }

  void _onBackToCamera(BuildContext context) {
    final appState = context.read<AppState>();
    appState.setCapturedImage(null);
    appState.resetSelection();
    appState.setStage(AppStage.camera);
    Navigator.pushReplacement(
      context,
      AppTransitions.fadeRoute(const CameraPage()),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXTRACTED WIDGETS — isolated rebuild scopes
// ═══════════════════════════════════════════════════════════════════════════════

/// Placeholder shown when no image is captured — const widget, never rebuilds
class _EmptyCanvasPlaceholder extends StatelessWidget {
  const _EmptyCanvasPlaceholder();

  @override
  Widget build(BuildContext context) {
    return Container(
      color: const Color(0xFF2C2C2E),
      child: const Center(
        child: Icon(Icons.image, color: Colors.white24, size: 80),
      ),
    );
  }
}

/// Top toolbar — separate widget so it never triggers canvas rebuilds
class _EditorTopToolbar extends StatelessWidget {
  final VoidCallback onBackToCamera;
  final VoidCallback onGoHome;

  const _EditorTopToolbar({
    required this.onBackToCamera,
    required this.onGoHome,
  });

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            _TopIconBtn('assets/icons/Vector.png', onTap: onBackToCamera),
            _TopIconBtn('assets/icons/home.png', onTap: onGoHome),
          ],
        ),
      ),
    );
  }
}

/// Icon button for the top toolbar
class _TopIconBtn extends StatelessWidget {
  final String assetPath;
  final VoidCallback onTap;

  const _TopIconBtn(this.assetPath, {required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 44,
        height: 44,
        decoration: const BoxDecoration(
          color: Colors.white12,
          shape: BoxShape.circle,
        ),
        child: Center(
          child: Image.asset(
            assetPath,
            width: 22,
            height: 22,
            color: Colors.white,
          ),
        ),
      ),
    );
  }
}

/// System icon button for the top toolbar
class _TopSysBtn extends StatelessWidget {
  final IconData icon;
  final bool filled;
  final VoidCallback onTap;

  const _TopSysBtn(this.icon, {this.filled = false, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: filled ? Colors.white24 : Colors.transparent,
          shape: BoxShape.circle,
        ),
        child: Icon(icon, color: Colors.white, size: 24),
      ),
    );
  }
}

/// Color preview circle — only rebuilds when selectedColor changes
class _ColorPreviewWidget extends StatelessWidget {
  const _ColorPreviewWidget();

  @override
  Widget build(BuildContext context) {
    final color = context.select<AppState, Color>((s) => s.selectedColor);
    return Container(
      width: 40,
      height: 40,
      decoration: BoxDecoration(color: color, shape: BoxShape.circle),
    );
  }
}

/// Bottom action button with icon and label
class _BottomAction extends StatelessWidget {
  final Widget child;
  final String label;
  final VoidCallback onTap;

  const _BottomAction({
    required this.child,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Padding(
            padding: const EdgeInsets.all(10.0),
            child: child,
          ),
          const SizedBox(height: 6),
          Text(
            label,
            style: const TextStyle(color: Colors.white70, fontSize: 13),
          ),
        ],
      ),
    );
  }
}

/// Icon in frame - for palette and material icons
class _IconInFrameWidget extends StatelessWidget {
  final String assetPath;
  final double size;

  const _IconInFrameWidget({required this.assetPath, this.size = 24});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: size,
      height: size,
      child: Stack(
        alignment: Alignment.center,
        children: [
          Image.asset(
            'assets/icons/ramka.png',
            width: size,
            height: size,
            fit: BoxFit.cover,
          ),
          Image.asset(
            assetPath,
            width: size * 0.65,
            height: size * 0.65,
            color: Colors.white,
          ),
        ],
      ),
    );
  }
}
