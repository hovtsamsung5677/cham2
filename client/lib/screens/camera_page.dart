import 'dart:typed_data';
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import '../models/app_state.dart';
import '../utils/transitions.dart';
import 'editor_screen.dart';
import 'projects_screen.dart';

Uint8List _fixImageOrientation(Uint8List bytes) {
  try {
    img.Image? decoded = img.decodeImage(bytes);
    if (decoded == null) return bytes;

    decoded = img.bakeOrientation(decoded);
    return Uint8List.fromList(img.encodeJpg(decoded));
  } catch (e) {
    debugPrint('Orientation fix error: $e');
    return bytes;
  }
}

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  CameraController? _cameraController;
  List<CameraDescription>? _cameras;
  bool _isCameraInitialized = false;
  int _currentCameraIndex = 0;
  double _selectedZoom = 1.0;
  bool _isFlashOn = false;
  Offset? _focusPoint;
  double _baseZoom = 1.0;
  Timer? _zoomDebounceTimer;

  Future<void> _initializeCamera([int? cameraIndex]) async {
    PermissionStatus cameraStatus = await Permission.camera.request();
    if (!cameraStatus.isGranted) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Камера требует разрешения')),
        );
      }
      return;
    }

    try {
      _cameras = await availableCameras();
      if (_cameras != null && _cameras!.isNotEmpty) {
        await _cameraController?.dispose();

        final index = cameraIndex ?? _currentCameraIndex;
        if (index >= _cameras!.length) {
          _currentCameraIndex = 0;
        } else {
          _currentCameraIndex = index;
        }

        _cameraController = CameraController(
          _cameras![_currentCameraIndex],
          ResolutionPreset.high,
          enableAudio: false,
        );
        await _cameraController!.initialize();

        try {
          await _cameraController!.setZoomLevel(_selectedZoom);
        } catch (e) {
          debugPrint('Error setting initial zoom: $e');
        }

        if (mounted) {
          setState(() {
            _isCameraInitialized = true;
          });
        }
      }
    } catch (e) {
      debugPrint('Camera initialization error: $e');
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Ошибка камеры: $e')));
      }
    }
  }

  Future<void> _switchCamera() async {
    if (_cameras == null || _cameras!.length < 2) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Переключение камеры недоступно')),
        );
      }
      return;
    }

    // Отключаем вспышку перед переключением камеры
    if (_isFlashOn &&
        _cameraController != null &&
        _cameraController!.value.isInitialized) {
      try {
        await _cameraController!.setFlashMode(FlashMode.off);
        _isFlashOn = false;
        if (mounted) setState(() {});
      } catch (e) {
        debugPrint('Error turning off flash: $e');
      }
    }

    final newIndex = (_currentCameraIndex + 1) % _cameras!.length;
    await _initializeCamera(newIndex);
  }

  Future<void> _takePicture() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    try {
      final XFile image = await _cameraController!.takePicture();
      var bytes = await image.readAsBytes();
      bytes = _fixImageOrientation(bytes);

      if (mounted) {
        context.read<AppState>().setCapturedImage(bytes);
        Navigator.push(
          context,
          AppTransitions.slideRoute(
            const EditorScreen(),
            direction: SlideDirection.left,
          ),
        );
      }
    } catch (e) {
      debugPrint('Error taking picture: $e');
    }
  }

  Future<void> _setZoom(double zoom) async {
    if (_cameraController != null && _cameraController!.value.isInitialized) {
      _zoomDebounceTimer?.cancel();
      _zoomDebounceTimer = Timer(const Duration(milliseconds: 50), () async {
        try {
          await _cameraController!.setZoomLevel(zoom);
        } catch (e) {
          debugPrint('Error setting zoom: $e');
        }
      });
    }
  }

  void _onCameraTap(TapDownDetails details) {
    if (_isCameraInitialized && _cameraController != null) {
      final Offset tapPosition = details.localPosition;
      final Size screenSize = MediaQuery.of(context).size;
      
      // Вычисляем точки фокусировки в пространстве камеры
      // CameraController.setFocusPoint accepts values normalized to 0-1
      final double focusX = (tapPosition.dx / screenSize.width).clamp(0.0, 1.0);
      final double focusY = (tapPosition.dy / screenSize.height).clamp(0.0, 1.0);
      
      // Устанавливаем точку фокусировки камеры
      _cameraController!.setFocusPoint(Offset(focusX, focusY));
      
      // Store the screen position for the focus frame
      setState(() {
        _focusPoint = tapPosition;
      });
      
      // Скрываем фокусную рамку через 2 секунды
      Future.delayed(const Duration(seconds: 2), () {
        if (mounted) {
          setState(() {
            _focusPoint = null;
          });
        }
      });
    }
  }

  Future<void> _toggleFlash() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    try {
      _isFlashOn = !_isFlashOn;
      await _cameraController!.setFlashMode(
        _isFlashOn ? FlashMode.always : FlashMode.off,
      );
      setState(() {});
    } catch (e) {
      debugPrint('Error toggling flash: $e');
    }
  }

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  @override
  void dispose() {
    if (_isFlashOn && _cameraController != null) {
      try {
        _cameraController!.setFlashMode(FlashMode.off);
      } catch (e) {
        debugPrint('Error turning off flash in dispose: $e');
      }
    }
    _cameraController?.dispose();
    _zoomDebounceTimer?.cancel();
    super.dispose();
  }

@override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF151412),
      body: Stack(
        children: [
          Center(
            child: AspectRatio(
              aspectRatio: 9 / 16,
              child: _isCameraInitialized && _cameraController != null
                  ? GestureDetector(
                      onScaleStart: (details) {
                        if (details.pointerCount == 2) {
                          _baseZoom = _selectedZoom;
                        }
                      },
                      onScaleUpdate: (details) {
                        if (details.pointerCount == 2 && details.scale != 1.0) {
                          final newZoom = (_baseZoom * details.scale).clamp(1.0, 4.0);
                          setState(() => _selectedZoom = newZoom);
                          _setZoom(newZoom);
                        }
                      },
                      child: CameraPreview(_cameraController!),
                    )
                  : const SizedBox.expand(),
            ),
          ),
          // Manual focus overlay - tappable anywhere for manual focus
          Positioned.fill(
            child: GestureDetector(
              onTapDown: (details) => _onCameraTap(details),
              behavior: HitTestBehavior.translucent,
              child: const SizedBox(),
            ),
          ),
          // Focus frame overlay - shows at tapped position when manual focus is active
          if (_focusPoint != null)
            Positioned(
              left: _focusPoint!.dx - 40,
              top: _focusPoint!.dy - 40,
              child: SizedBox(
                width: 80,
                height: 80,
                child: CustomPaint(painter: _FocusFramePainter()),
              ),
            ),
          // Top buttons overlay
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  _TopButton(
                    iconPath: 'assets/icons/home.png',
                    onTap: () async {
                      if (_isFlashOn &&
                          _cameraController != null &&
                          _cameraController!.value.isInitialized) {
                        try {
                          await _cameraController!.setFlashMode(FlashMode.off);
                          _isFlashOn = false;
                          if (mounted) setState(() {});
                        } catch (e) {
                          debugPrint('Error turning off flash: $e');
                        }
                      }
                      if (!mounted) return;
                      Navigator.pushAndRemoveUntil(
                        context,
                        AppTransitions.fadeRoute(const ProjectsScreen()),
                        (route) => false,
                      );
                    },
                  ),
                  _TopButton(
                    iconPath: 'assets/icons/light.png',
                    onTap: _toggleFlash,
                    isActive: _isFlashOn,
                  ),
                ],
              ),
            ),
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: Container(
              color: const Color(0xFF151412),
              padding: const EdgeInsets.only(bottom: 32, top: 16),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1C1C1C),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      '${_selectedZoom.toStringAsFixed(1)}x',
                      style: const TextStyle(
                        color: Color(0xFFF5A623),
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  const SizedBox(height: 20),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      GestureDetector(
                        onTap: _pickFromGallery,
                        child: Container(
                          width: 52,
                          height: 52,
                          decoration: BoxDecoration(
                            color: const Color(0xFF404040),
                            borderRadius: BorderRadius.circular(14),
                          ),
                          child: Center(
                            child: Image.asset(
                              'assets/icons/Add_Image.png',
                              width: 26,
                              height: 26,
                            ),
                          ),
                        ),
                      ),
                      GestureDetector(
                        onTap: _takePicture,
                        child: Container(
                          width: 72,
                          height: 72,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: const Color(0xFFF5A623),
                            boxShadow: [
                              BoxShadow(
                                color: const Color(0xFFF5A623).withValues(alpha: 0.45),
                                blurRadius: 20,
                                spreadRadius: 2,
                              ),
                            ],
                          ),
                        ),
                      ),
                      GestureDetector(
                        onTap: _switchCamera,
                        child: Container(
                          width: 52,
                          height: 52,
                          decoration: BoxDecoration(
                            color: const Color(0xFF404040),
                            borderRadius: BorderRadius.circular(14),
                          ),
child: Center(
                             child: Image.asset(
                               'assets/icons/frontalka.png',
                               width: 26,
                               height: 26,
                             ),
                           ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _pickFromGallery() async {
    PermissionStatus status;
    if (Theme.of(context).platform == TargetPlatform.iOS) {
      status = await Permission.photos.request();
    } else {
      // Android: используем Permission.photos для Android 13+ (API 33+), иначе storage
      try {
        status = await Permission.photos.request();
        if (!status.isGranted) {
          status = await Permission.storage.request();
        }
      } catch (e) {
        // Если Permission.photos не поддерживается (Android <13), используем storage
        status = await Permission.storage.request();
      }
    }

    if (!status.isGranted) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Требуется разрешение доступа к галерее'),
          ),
        );
      }
      return;
    }

    try {
      // Отключаем вспышку перед открытием галереи
      if (_isFlashOn &&
          _cameraController != null &&
          _cameraController!.value.isInitialized) {
        await _cameraController!.setFlashMode(FlashMode.off);
        _isFlashOn = false;
        if (mounted) setState(() {});
      }

      final XFile? image = await ImagePicker().pickImage(
        source: ImageSource.gallery,
        maxWidth: 1920,
        imageQuality: 100,
      );
      if (image != null) {
        var bytes = await image.readAsBytes();
        bytes = _fixImageOrientation(bytes);
        debugPrint('Picked image from gallery: ${bytes.length} bytes');

        if (mounted) {
          // Store image in AppState
          context.read<AppState>().setCapturedImage(bytes);
          // Navigate to editor with slide transition
          Navigator.push(
            context,
            AppTransitions.slideRoute(
              const EditorScreen(),
              direction: SlideDirection.left,
            ),
          );
        }
      } else {
        debugPrint('No image selected from gallery');
      }
    } catch (e) {
      debugPrint('Error picking image: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Ошибка выбора изображения: $e')),
        );
      }
    }
  }
}

class _TopButton extends StatelessWidget {
  final String iconPath;
  final VoidCallback? onTap;
  final bool isActive;
  const _TopButton({
    required this.iconPath,
    this.onTap,
    this.isActive = false,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          color: isActive ? const Color(0xFFFFC107) : Colors.white12,
          shape: BoxShape.circle,
          border: isActive ? Border.all(color: Colors.white, width: 2) : null,
        ),
        child: Center(
          child: Image.asset(
            iconPath,
            width: 22,
            height: 22,
            color: isActive ? Colors.black : Colors.white,
          ),
        ),
      ),
    );
  }
}

class _FocusFramePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    const cornerLen = 16.0;
    final w = size.width;
    final h = size.height;

    final paths = [
      [Offset(0, cornerLen), Offset(0, 0), Offset(cornerLen, 0)],
      [Offset(w - cornerLen, 0), Offset(w, 0), Offset(w, cornerLen)],
      [Offset(0, h - cornerLen), Offset(0, h), Offset(cornerLen, h)],
      [Offset(w - cornerLen, h), Offset(w, h), Offset(w, h - cornerLen)],
    ];

    for (final pts in paths) {
      final path = Path()
        ..moveTo(pts[0].dx, pts[0].dy)
        ..lineTo(pts[1].dx, pts[1].dy)
        ..lineTo(pts[2].dx, pts[2].dy);
      canvas.drawPath(path, paint);
    }
  }

  @override
  bool shouldRepaint(_) => false;
}
