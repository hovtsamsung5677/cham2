import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

/// Прозрачный зацикленный экран загрузки.
///
/// Проигрывает WebM с альфа-каналом (VP8/yuva420p). На iOS, где системный
/// плеер может не поддерживать WebM, выполняется fallback на MOV
/// (ProRes 4444 с альфой). Видео центрируется поверх затемнённого фона и
/// зацикливается, пока [visible] == true.
class VideoLoadingOverlay extends StatefulWidget {
  final bool visible;
  final String? message;

  const VideoLoadingOverlay({
    super.key,
    required this.visible,
    this.message,
  });

  @override
  State<VideoLoadingOverlay> createState() => _VideoLoadingOverlayState();
}

class _VideoLoadingOverlayState extends State<VideoLoadingOverlay> {
  VideoPlayerController? _controller;
  bool _useWebm = true;

  @override
  void initState() {
    super.initState();
    _initController();
  }

  @override
  void didUpdateWidget(covariant VideoLoadingOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.visible != widget.visible) {
      _applyPlayState();
    }
  }

  Future<void> _initController() async {
    if (!widget.visible) return;

    final asset = _useWebm ? 'assets/zagruuuuzka.webm' : 'assets/zagruuuuzka.mov';
    final controller = VideoPlayerController.asset(
      asset,
      videoPlayerOptions: VideoPlayerOptions(mixWithOthers: true),
    );

    try {
      await controller.initialize();
      if (!mounted || !widget.visible) {
        await controller.dispose();
        return;
      }
      await controller.setLooping(true);
      await controller.setVolume(0.0);
      setState(() {
        _controller = controller;
      });
      _applyPlayState();
    } catch (e) {
      await controller.dispose();
      if (_useWebm && !kIsWeb) {
        // WebM не поддерживается на этой платформе (напр. iOS) — пробуем MOV.
        _useWebm = false;
        await _initController();
      }
    }
  }

  void _applyPlayState() {
    final controller = _controller;
    if (controller == null) return;
    if (widget.visible) {
      controller.play();
    } else {
      controller.pause();
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.visible) return const SizedBox.shrink();

    return Container(
      // Затемнённый фон поверх экрана.
      color: Colors.black54,
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _buildVideo(),
            if (widget.message != null) ...[
              const SizedBox(height: 14),
              Text(
                widget.message!,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildVideo() {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      // Запасной вариант, пока видео грузится или при ошибке обоих форматов.
      return const SizedBox(
        width: 220,
        height: 220,
        child: Center(
          child: CircularProgressIndicator(color: Color(0xFFF5C518)),
        ),
      );
    }

    return SizedBox(
      width: 240,
      height: 240,
      child: VideoPlayer(controller),
    );
  }
}
