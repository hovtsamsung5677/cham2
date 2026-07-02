import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:gallery_saver/gallery_saver.dart';
import 'package:provider/provider.dart';
import 'package:permission_handler/permission_handler.dart';
import '../models/app_state.dart';
import '../utils/transitions.dart';
import 'projects_screen.dart';

class ExportScreen extends StatefulWidget {
  final Uint8List? initialImageBytes;

  const ExportScreen({super.key, this.initialImageBytes});

  @override
  State<ExportScreen> createState() => _ExportScreenState();
}

class _ExportScreenState extends State<ExportScreen> {
  // Флаг удержания кнопки сравнения
  bool _isCompareHeld = false;

  @override
  Widget build(BuildContext context) {
    final capturedImage = context.select<AppState, Uint8List?>((s) => s.capturedImage);
    final previewImage = context.select<AppState, Uint8List?>((s) => s.previewImage);
    final displayImage = _isCompareHeld && capturedImage != null
        ? capturedImage
        : (widget.initialImageBytes ?? previewImage ?? capturedImage);

    return Scaffold(
      backgroundColor: const Color(0xFF151412),
      appBar: AppBar(
        backgroundColor: const Color(0xFF151412),
        foregroundColor: Colors.white,
        title: const Text('Результат'),
        leading: GestureDetector(
          onTap: () => Navigator.pop(context),
          child: Container(
            width: 44,
            height: 44,
            margin: const EdgeInsets.only(left: 8),
            decoration: const BoxDecoration(
              color: Colors.white12,
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.arrow_back, color: Colors.white, size: 22),
          ),
        ),
actions: [
            GestureDetector(
              onTap: () {
                context.read<AppState>().setCapturedImage(null);
                Navigator.pushAndRemoveUntil(
                  context,
                  AppTransitions.fadeRoute(const ProjectsScreen()),
                  (route) => false,
                );
              },
              child: Container(
                width: 44,
                height: 44,
                margin: const EdgeInsets.only(right: 8),
                decoration: const BoxDecoration(
                  color: Colors.white12,
                  shape: BoxShape.circle,
                ),
                child: Center(
                  child: Image.asset(
                    'assets/icons/home.png',
                    width: 22,
                    height: 22,
                    color: Colors.white,
                  ),
                ),
              ),
            ),
          ],
      ),
      body: Column(
        children: [
          Expanded(
            child: _buildImageDisplay(displayImage),
          ),
          Container(
            color: const Color(0xFF151412),
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                _buildCompareButton(),
                const SizedBox(height: 12),
                SizedBox(
                  width: double.infinity,
                  height: 54,
                  child: ElevatedButton.icon(
                    onPressed: () => _saveImage(context, displayImage),
                    icon: const Icon(Icons.download, size: 24),
                    label: const Text(
                      'Скачать',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFFF5C518),
                      foregroundColor: Colors.black,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(28),
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                SizedBox(
                  width: double.infinity,
                  height: 54,
                  child: ElevatedButton.icon(
                    onPressed: () => _shareImage(context, displayImage),
                    icon: const Icon(Icons.share, size: 24),
                    label: const Text(
                      'Отправить',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF404040),
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(28),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImageDisplay(Uint8List? displayImage) {
    if (displayImage == null) {
      return const Center(
        child: Text('Нет изображения', style: TextStyle(color: Colors.white)),
      );
    }

    final imageProvider = MemoryImage(displayImage);

    return InteractiveViewer(
      minScale: 1.0,
      maxScale: 4.0,
      child: Center(
        child: Image(
          image: imageProvider,
          fit: BoxFit.contain,
          gaplessPlayback: true,
          frameBuilder: (context, child, frame, wasSynchronouslyLoaded) {
            if (wasSynchronouslyLoaded) return child;
            return AnimatedOpacity(
              opacity: frame == null ? 0 : 1,
              duration: const Duration(milliseconds: 200),
              curve: Curves.easeOut,
              child: child,
            );
          },
        ),
      ),
    );
  }

  Widget _buildCompareButton() {
    final hasOriginal = context.select<AppState, Uint8List?>((s) => s.capturedImage) != null;
    final hasRecolored = context.select<AppState, Uint8List?>((s) => s.previewImage) != null;

    if (!hasOriginal || !hasRecolored) {
      return const SizedBox.shrink();
    }

    return SizedBox(
      width: double.infinity,
      height: 54,
      child: GestureDetector(
        onTapDown: (_) => setState(() => _isCompareHeld = true),
        onTapUp: (_) => setState(() => _isCompareHeld = false),
        onTapCancel: () => setState(() => _isCompareHeld = false),
        onLongPressStart: (_) => setState(() => _isCompareHeld = true),
        onLongPressEnd: (_) => setState(() => _isCompareHeld = false),
        child: Container(
          decoration: BoxDecoration(
            color: _isCompareHeld ? const Color(0xFFFFC107) : const Color(0xFF404040),
            borderRadius: BorderRadius.circular(28),
            border: Border.all(
              color: _isCompareHeld ? Colors.white : Colors.grey,
              width: 2,
            ),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.compare_arrows, color: Colors.white, size: 24),
              const SizedBox(width: 8),
              Text(
                _isCompareHeld ? 'Оригинал' : 'Перекраска',
                style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _saveImage(BuildContext context, Uint8List? imageBytes) async {
    if (imageBytes == null) return;

    try {
      if (Platform.isIOS) {
        final status = await Permission.photosAddOnly.request();
        if (status != PermissionStatus.granted && status != PermissionStatus.limited) {
          if (context.mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('Нет разрешения на сохранение в галерею')),
            );
          }
          return;
        }
      }

      final directory = await getTemporaryDirectory();
      final fileName = 'recolored_${DateTime.now().millisecondsSinceEpoch}.png';
      final file = File('${directory.path}/$fileName');
      await file.writeAsBytes(imageBytes);

      final result = await GallerySaver.saveImage(
        file.path,
        albumName: 'Furniture Recoloring',
      );

      if (context.mounted) {
        if (result == true) {
          context.read<AppState>().addProject(imageBytes);
          if (!mounted) return;
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Фото сохранено в галерее')),
          );
        } else {
          if (!mounted) return;
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Ошибка сохранения в галерее')),
          );
        }
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Ошибка сохранения: $e')),
        );
      }
    }
  }

  Future<void> _shareImage(BuildContext context, Uint8List? imageBytes) async {
    if (imageBytes == null) return;

    try {
      final fileName = 'recolored_share_${DateTime.now().millisecondsSinceEpoch}.png';
      final xFile = XFile.fromData(
        imageBytes,
        name: fileName,
        mimeType: 'image/png',
        lastModified: DateTime.now(),
      );
      await SharePlus.instance.share(
        ShareParams(
          files: [xFile],
          text: 'Посмотри на моё перекрашенное фото!',
        ),
      );
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Ошибка отправки: $e')),
        );
      }
    }
  }
}
