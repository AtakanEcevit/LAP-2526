(() => {
    const LOCALE = 'tr-TR';

    const MESSAGES = {
        'meta.title': 'FaceVerify Campus',
        'meta.description': 'FaceVerify Campus sınav kimliği doğrulama demosu.',
        'confirm.resetDemo': 'FaceVerify Campus demo verileri sıfırlansın mı?',
        'confirm.submitVerification': 'Doğrulama gönderilsin mi?\n\nÖğrenci: {student}\nSınav: {exam}\nGörüntü kaynağı: {source}\n\nBu işlem yeni bir doğrulama denemesi oluşturur.',
        'source.preloaded': 'hazır sentetik FLUXSynID demo selfiesi',
        'source.uploaded': 'yüklenen selfie',
        'source.uploadedNamed': 'yüklenen selfie ({filename})',
        'context.summary': '{university} · {scenario} · {model} · Eşik {threshold}',
        'gate.step': 'Adım {step} / 4',
        'queue.summary': '{review} inceleme bekliyor | {verified} doğrulandı | {waiting} bekliyor | {blocked} engellendi',
        'sparkline.trend': '{label} eğilimi',
        'policy.title': 'Bu sınav erişimden önce {model} doğrulaması gerektirir.',
        'policy.detail': 'Eşik {threshold}. Yüksek skor daha güçlü eşleşme anlamına gelir. Doğrulanamayan öğrenciler yedek kimlik incelemesi isteyebilir.',
        'policy.adminDetail': 'Model bazlı kayıt gereklidir.',
    };

    const STATUS_LABELS = {
        'Verified': 'Doğrulandı',
        'Manual Review': 'Manuel İnceleme',
        'Rejected': 'Reddedildi',
        'Approved by Proctor': 'Gözetmen Onayladı',
        'Fallback Requested': 'Yedek Kontrol İstendi',
        'Enrollment Needed': 'Kayıt Gerekli',
        'Not Started': 'Başlamadı',
        'Pending Verification': 'Doğrulama Bekliyor',
        'Waiting': 'Bekliyor',
        'Ready': 'Hazır',
        'Partial Enrollment': 'Eksik Kayıt',
        'No Enrollment': 'Kayıt Yok',
        'Model Mismatch': 'Model Uyumsuzluğu',
        'Access Granted': 'Erişim Verildi',
        'Needs Review': 'İnceleme Gerekli',
        'Flagged / Denied': 'İşaretli / Reddedildi',
    };

    const DECISION_LABELS = {
        verified: 'Doğrulandı',
        manual_review: 'Manuel İnceleme',
        rejected: 'Reddedildi',
        none: 'Deneme yok',
        Pass: 'Geçti',
        Review: 'İnceleme',
    };

    const EVENT_LABELS = {
        demo_reset: 'Demo sıfırlandı',
        course_saved: 'Ders kaydedildi',
        exam_saved: 'Sınav kaydedildi',
        roster_imported: 'Liste içe aktarıldı',
        student_enrolled: 'Öğrenci kaydedildi',
        student_preuploaded: 'Öğrenci ön yüklendi',
        verification_attempted: 'Doğrulama denendi',
        manual_review_completed: 'Manuel inceleme tamamlandı',
        event: 'olay',
    };

    const STATIC_TEXT = {
        'Secure Exam Platform': 'Güvenli Sınav Platformu',
        'Main': 'Ana Menü',
        'Dashboard': 'Gösterge Paneli',
        'Exam Gate': 'Sınav Girişi',
        'Live Monitor': 'Canlı İzleme',
        'Exam Sessions': 'Sınav Oturumları',
        'Review & Actions': 'İnceleme ve İşlemler',
        'Pending Review': 'İnceleme Bekleyenler',
        'Flagged Attempts': 'İşaretli Denemeler',
        'Audit Logs': 'Denetim Kayıtları',
        'Settings': 'Ayarlar',
        'Exam Settings': 'Sınav Ayarları',
        'Verification Rules': 'Doğrulama Kuralları',
        'System Status': 'Sistem Durumu',
        'Connecting': 'Bağlanıyor',
        'Connection interrupted': 'Bağlantı kesildi',
        'All systems operational': 'Tüm sistemler çalışıyor',
        'Loading date': 'Tarih yükleniyor',
        'Atılım University Demo': 'Atılım Üniversitesi Demo',
        'AtÄ±lÄ±m University Demo': 'Atılım Üniversitesi Demo',
        'AtÄ±lÄ±m-styled campus scenario': 'Atılım tarzı kampüs senaryosu',
        'Atilim University Demo - SE 204 Midterm - Hybrid FaceNet - Threshold 0.300': 'Atılım Üniversitesi Demo - SE 204 Ara Sınav - Hybrid FaceNet - Eşik 0.300',
        'Atilim University Demo - SE 204 Midterm - FaceNet Contrastive Proto - Threshold 0.800884': 'Atılım Üniversitesi Demo - SE 204 Ara Sınav - FaceNet Contrastive Proto - Eşik 0.800884',
        'Biometric Exam Access Verification': 'Biyometrik Sınav Erişim Doğrulaması',
        'Live Monitoring': 'Canlı İzleme',
        'Instructor': 'Öğretim Görevlisi',
        'Notifications': 'Bildirimler',
        'Help': 'Yardım',
        'Active exam policy': 'Aktif sınav politikası',
        'Details': 'Ayrıntılar',
        'University': 'Üniversite',
        'Course': 'Ders',
        'Exam': 'Sınav',
        'Model': 'Model',
        'Threshold': 'Eşik',
        'Window': 'Zaman Aralığı',
        'Persona': 'Rol',
        'Demo Operator': 'Demo Operatörü',
        'Live Simulation': 'Canlı Simülasyon',
        'Student: Aylin Kaya': 'Öğrenci: Aylin Kaya',
        'Review Desk: Proctor Lee': 'İnceleme Masası: Proctor Lee',
        'Operations: Registrar Admin': 'Operasyonlar: Kayıt Yetkilisi',
        'Operations: Model Evidence': 'Operasyonlar: Model Kanıtı',
        'Calibration pending': 'Kalibrasyon bekliyor',
        'Workspace views': 'Çalışma alanı görünümleri',
        'Simulation': 'Simülasyon',
        'Review Desk': 'İnceleme Masası',
        'Operations': 'Operasyonlar',
        'Demo scenario': 'Demo senaryosu',
        'Policy': 'Politika',
        'Live Sim': 'Canlı Sim',
        'Student Gate': 'Öğrenci Girişi',
        'Audit/Evidence': 'Denetim/Kanıt',
        'SE 204 exam identity gate': 'SE 204 sınav kimlik kapısı',
        'A focused walkthrough: set the exam policy, enroll Aylin, verify access, trigger review, and show the audit trail.': 'Odaklı akış: sınav politikasını belirleyin, Aylin’i kaydedin, erişimi doğrulayın, incelemeyi tetikleyin ve denetim izini gösterin.',
        'Start Demo': 'Demoyu Başlat',
        'Confirm SE 204 exam policy': 'SE 204 sınav politikasını onayla',
        "Enroll Aylin's face samples": 'Aylin’in yüz örneklerini kaydet',
        'Verify matching student access': 'Eşleşen öğrenci erişimini doğrula',
        'Trigger wrong-face review path': 'Yanlış yüz inceleme yolunu tetikle',
        'Approve, deny, or export evidence': 'Onayla, reddet veya kanıtı dışa aktar',
        'Jump to step': 'Adıma git',
        'Policy and roster': 'Politika ve liste',
        'Split-screen simulation': 'Bölünmüş ekran simülasyonu',
        'Audit and evidence': 'Denetim ve kanıt',
        'Open Step': 'Adımı Aç',
        'Demo data': 'Demo verileri',
        'Student': 'Öğrenci',
        'Proctor': 'Gözetmen',
        'Admin': 'Yönetici',
        'Review desk': 'İnceleme masası',
        'Registrar Admin': 'Kayıt Yetkilisi',
        'SE 204 owner': 'SE 204 sorumlusu',
        'Roster': 'Liste',
        'Access Granted': 'Erişim Verildi',
        'This showcase is consent-based 1:1 exam access verification. It does not claim surveillance, cheating detection, or legal identity proof.': 'Bu demo, rızaya dayalı bire bir sınav erişim doğrulamasıdır. Gözetim, kopya tespiti veya yasal kimlik kanıtı iddiası taşımaz.',
        'Student and instructor POV': 'Öğrenci ve öğretim görevlisi görünümü',
        'Refresh': 'Yenile',
        'Simulation scenarios': 'Simülasyon senaryoları',
        'Matching selfie': 'Eşleşen selfie',
        'Needs review': 'İnceleme gerekli',
        'No enrollment': 'Kayıt yok',
        'Model mismatch': 'Model uyumsuzluğu',
        'Fallback requested': 'Yedek kontrol istendi',
        'Simulation point of view': 'Simülasyon bakış açısı',
        'Student POV': 'Öğrenci Görünümü',
        'Instructor/Admin POV': 'Öğretim Görevlisi/Yönetici Görünümü',
        'Student point of view': 'Öğrenci bakış açısı',
        'STUDENT SCREEN': 'ÖĞRENCİ EKRANI',
        'What the exam taker sees': 'Sınava girenin gördüğü ekran',
        'Student Point of View': 'Öğrenci Bakış Açısı',
        'Exam access gate': 'Sınav erişim kapısı',
        'Switch Student': 'Öğrenci Değiştir',
        'Choose who appears in both POVs': 'İki görünümde görünecek öğrenciyi seçin',
        'Select a student': 'Öğrenci seçin',
        'Midterm Exam access gate': 'Ara Sınav erişim kapısı',
        'Consent': 'Rıza',
        'Enrollment': 'Kayıt',
        'Verification': 'Doğrulama',
        'Result': 'Sonuç',
        'Student confirms consent before any capture.': 'Öğrenci herhangi bir çekimden önce rızasını onaylar.',
        'Checks model-specific enrollment readiness.': 'Modele özel kayıt hazır olup olmadığını kontrol eder.',
        'Submits the exam-day selfie to the active policy.': 'Sınav günü selfiesini aktif politikaya gönderir.',
        'Shows access, review, fallback, or blocked outcome.': 'Erişim, inceleme, yedek kontrol veya engelleme sonucunu gösterir.',
        'I consent to this exam access check.': 'Bu sınav erişim kontrolüne rıza veriyorum.',
        'Select an exam and student to begin.': 'Başlamak için sınav ve öğrenci seçin.',
        'Student live camera frame': 'Öğrenci canlı kamera alanı',
        'Camera preview appears after verification': 'Kamera önizlemesi doğrulamadan sonra görünür',
        'Please hold still...': 'Lütfen sabit durun...',
        'Exam-day selfie': 'Sınav günü selfiesi',
        'Verify for Exam Access': 'Sınav Erişimi İçin Doğrula',
        'Load Preloaded Selfie': 'Hazır Selfieyi Yükle',
        'Student Message': 'Öğrenci Mesajı',
        'No attempt yet': 'Henüz deneme yok',
        'The student has not submitted an exam-day selfie in this simulation.': 'Öğrenci bu simülasyonda henüz sınav günü selfiesi göndermedi.',
        'Score': 'Skor',
        'Attempt': 'Deneme',
        'same exam event': 'aynı sınav olayı',
        'Instructor and admin point of view': 'Öğretim görevlisi ve yönetici bakış açısı',
        'INSTRUCTOR / ADMIN SCREEN': 'ÖĞRETİM GÖREVLİSİ / YÖNETİCİ EKRANI',
        'What staff monitors and decides': 'Personelin izlediği ve karar verdiği ekran',
        'Instructor/Admin Point of View': 'Öğretim Görevlisi/Yönetici Bakış Açısı',
        'Exam operations cockpit': 'Sınav operasyon paneli',
        'Instructor admin control center metrics': 'Öğretim görevlisi yönetim merkezi metrikleri',
        'Total Attempts': 'Toplam Deneme',
        'Live': 'Canlı',
        'Verified': 'Doğrulandı',
        'Flagged / Denied': 'İşaretli / Reddedildi',
        'No Enrollment': 'Kayıt Yok',
        'Live Verification Feed': 'Canlı Doğrulama Akışı',
        'View All': 'Tümünü Gör',
        'Status': 'Durum',
        'Match Score': 'Eşleşme Skoru',
        'Actions': 'İşlemler',
        'Captured Photo': 'Çekilen Fotoğraf',
        'No image': 'Görüntü yok',
        'Enrolled Photo': 'Kayıtlı Fotoğraf',
        'Decision': 'Karar',
        'Warnings': 'Uyarılar',
        'Attempt Time': 'Deneme Zamanı',
        'Reviewer': 'İnceleyen',
        'Reason': 'Gerekçe',
        'Manual ID check completed.': 'Manuel kimlik kontrolü tamamlandı.',
        'Approve Access': 'Erişimi Onayla',
        'Deny Access': 'Erişimi Reddet',
        'Mark for Review': 'İncelemeye İşaretle',
        'Verification Audit Trail': 'Doğrulama Denetim İzi',
        'Selected attempt event path': 'Seçili denemenin olay akışı',
        'View Full Log': 'Tam Kaydı Gör',
        'Simulation timeline': 'Simülasyon zaman çizelgesi',
        'Live event timeline': 'Canlı olay zaman çizelgesi',
        'Export CSV': 'CSV Dışa Aktar',
        'Student access check': 'Öğrenci erişim kontrolü',
        'Switch student or exam': 'Öğrenci veya sınav değiştir',
        'Student verification progress': 'Öğrenci doğrulama ilerlemesi',
        'Consent gates the verification flow.': 'Rıza, doğrulama akışını başlatır.',
        'Enrollment confirms usable face samples for the model.': 'Kayıt, model için kullanılabilir yüz örneklerini doğrular.',
        'Exam-day selfie is checked against the active threshold.': 'Sınav günü selfiesi aktif eşiğe göre kontrol edilir.',
        'Final access decision is shown to the student.': 'Nihai erişim kararı öğrenciye gösterilir.',
        'Step 1': 'Adım 1',
        'Step 2': 'Adım 2',
        'Step 3': 'Adım 3',
        'Consent to one exam access check': 'Tek seferlik sınav erişim kontrolüne rıza',
        'Face verification is used only to confirm the student starting this exam matches the enrolled student profile.': 'Yüz doğrulama yalnızca sınava başlayan öğrencinin kayıtlı öğrenci profiliyle eşleştiğini doğrulamak için kullanılır.',
        'I consent to a one-time face verification for this exam session.': 'Bu sınav oturumu için tek seferlik yüz doğrulamasına rıza veriyorum.',
        'Continue to Enrollment': 'Kayda Devam Et',
        'Enroll face samples': 'Yüz örneklerini kaydet',
        'Enrollment Status': 'Kayıt Durumu',
        'Model-specific enrollment: -': 'Modele özel kayıt: -',
        'Recommended': 'Önerilen',
        'Slots Left': 'Kalan Hak',
        'Add 3 clear face samples before verification.': 'Doğrulamadan önce 3 net yüz örneği ekleyin.',
        'Add clear front-facing samples with small changes in expression or lighting.': 'İfade veya ışıkta küçük değişiklikler olan net, önden çekilmiş örnekler ekleyin.',
        'No files selected.': 'Dosya seçilmedi.',
        'Add Face Samples': 'Yüz Örnekleri Ekle',
        'Continue to Verification': 'Doğrulamaya Devam Et',
        'Submit exam-day selfie': 'Sınav günü selfiesini gönder',
        'Upload the image used for this exam access attempt.': 'Bu sınav erişim denemesinde kullanılacak görüntüyü yükleyin.',
        'Demo tools': 'Demo araçları',
        'Preloaded Demo Selfie': 'Hazır Demo Selfiesi',
        'Stage the matching synthetic selfie': 'Eşleşen sentetik selfieyi hazırla',
        "Loads the student's reserved FLUXSynID selfie for manual confirmation. It does not submit verification until you press Verify for Exam Access.": 'Öğrencinin ayrılmış FLUXSynID selfiesini manuel onay için yükler. Sınav Erişimi İçin Doğrula düğmesine basılana kadar doğrulama gönderilmez.',
        'Wrong-Face Demo Path': 'Yanlış Yüz Demo Yolu',
        'Need to prove Manual Review?': 'Manuel incelemeyi göstermek mi gerekiyor?',
        'Submit a selfie from a different person for Aylin Kaya. The attempt should land in Manual Review or Rejected, then appear in Review Desk.': 'Aylin Kaya için farklı bir kişiye ait selfie gönderin. Deneme Manuel İnceleme veya Reddedildi durumuna düşer ve İnceleme Masası’nda görünür.',
        'Open Review Desk After Attempt': 'Denemeden Sonra İnceleme Masasını Aç',
        'Select a student, consent, and submit a selfie.': 'Öğrenci seçin, rıza verin ve selfie gönderin.',
        'Verification ID': 'Doğrulama ID',
        'Timestamp': 'Zaman Damgası',
        'Student readiness summary': 'Öğrenci hazırlık özeti',
        'Gate Context': 'Giriş Bağlamı',
        'Active Exam Policy': 'Aktif Sınav Politikası',
        'Face verification required before exam access.': 'Sınav erişiminden önce yüz doğrulama gereklidir.',
        'Threshold and model details update with the selected exam.': 'Eşik ve model ayrıntıları seçili sınava göre güncellenir.',
        'Last Attempt': 'Son Deneme',
        'Download Matching Selfie': 'Eşleşen Selfieyi İndir',
        'Needs review queue': 'İnceleme kuyruğu',
        'Refresh Roster': 'Listeyi Yenile',
        'Roster status summary': 'Liste durum özeti',
        'Only action-needed students are shown first. Expand filters to inspect the full roster.': 'Önce işlem gerektiren öğrenciler gösterilir. Tüm listeyi incelemek için filtreleri genişletin.',
        'Review queue metrics': 'İnceleme kuyruğu metrikleri',
        'Show full roster and filters': 'Tüm listeyi ve filtreleri göster',
        'Roster filters': 'Liste filtreleri',
        'All statuses': 'Tüm durumlar',
        'All decisions': 'Tüm kararlar',
        'No attempt': 'Deneme yok',
        'Student name': 'Öğrenci adı',
        'Student ID': 'Öğrenci ID',
        'Sort': 'Sırala',
        'Needs review first': 'Önce inceleme gerekenler',
        'Name A-Z': 'Ad A-Z',
        'Score low-high': 'Skor düşük-yüksek',
        'Action': 'İşlem',
        'Review Drawer': 'İnceleme Paneli',
        'Select an attempt': 'Deneme seçin',
        'Select an attempt from the Needs Review queue or roster.': 'İnceleme kuyruğundan veya listeden bir deneme seçin.',
        'Enrolled Reference': 'Kayıtlı Referans',
        'Exam Attempt': 'Sınav Denemesi',
        'Source': 'Kaynak',
        'Previous attempts': 'Önceki denemeler',
        'Approve': 'Onayla',
        'Deny': 'Reddet',
        'Request Fallback': 'Yedek Kontrol İste',
        'Policy, roster, audit, and evidence': 'Politika, liste, denetim ve kanıt',
        'Exam policy': 'Sınav politikası',
        'Current Policy Controls Every Screen': 'Geçerli Politika Tüm Ekranları Yönetir',
        'Changing model type requires students enrolled under another model to re-enroll.': 'Model türünü değiştirmek, başka modelle kayıtlı öğrencilerin yeniden kaydolmasını gerektirir.',
        'Exam ID': 'Sınav ID',
        'Exam Name': 'Sınav Adı',
        'Exam window': 'Sınav zamanı',
        'Start': 'Başlangıç',
        'End': 'Bitiş',
        'Save Exam Policy': 'Sınav Politikasını Kaydet',
        'Course details': 'Ders ayrıntıları',
        'Course ID': 'Ders ID',
        'Term': 'Dönem',
        'Course Name': 'Ders Adı',
        'Save Course': 'Dersi Kaydet',
        'FLUXSynID preupload': 'FLUXSynID ön yükleme',
        'Seed the simulation roster with synthetic student faces and model-backed enrollments.': 'Simülasyon listesini sentetik öğrenci yüzleri ve model destekli kayıtlarla doldurun.',
        'Dataset path': 'Veri seti yolu',
        'Count': 'Sayı',
        'Seed': 'Tohum',
        'Checking FLUXSynID dataset...': 'FLUXSynID veri seti kontrol ediliyor...',
        'Resolved path': 'Çözümlenen yol',
        'Test kit path': 'Test paketi yolu',
        'Exported selfies': 'Dışa aktarılan selfieler',
        '0 matching selfies exported': '0 eşleşen selfie dışa aktarıldı',
        'Preupload Student Faces': 'Öğrenci Yüzlerini Ön Yükle',
        'Export Test Selfies': 'Test Selfielerini Dışa Aktar',
        'Download Test Set ZIP': 'Test Seti ZIP İndir',
        'CSV columns: student_id,name,email. Duplicate rows update the existing demo roster entry.': 'CSV sütunları: student_id,name,email. Yinelenen satırlar mevcut demo liste kaydını günceller.',
        'Import CSV': 'CSV İçe Aktar',
        'Audit/export': 'Denetim/dışa aktarım',
        'Audit Log': 'Denetim Kaydı',
        'Time': 'Zaman',
        'Event': 'Olay',
        'Actor': 'Aktör',
        'Message': 'Mesaj',
        'Model evidence': 'Model kanıtı',
        'Face Image A': 'Yüz Görüntüsü A',
        'Face Image B': 'Yüz Görüntüsü B',
        'Compare Faces': 'Yüzleri Karşılaştır',
        'Model Output': 'Model Çıktısı',
        'Waiting for images': 'Görüntüler bekleniyor',
        'Calibration': 'Kalibrasyon',
        'Report-only workflow available': 'Yalnızca raporlama akışı mevcut',
        'Model notes': 'Model notları',
        'Higher score means stronger match.': 'Yüksek skor daha güçlü eşleşme anlamına gelir.',
        'Hybrid FaceNet, FaceNet Proto, and FaceNet Contrastive Proto use RGB 160x160 input and a 512-d normalized embedding. The current runtime threshold is an initial setting until target-data calibration supports a production value.': 'Hybrid FaceNet, FaceNet Proto ve FaceNet Contrastive Proto, RGB 160x160 girdi ve 512 boyutlu normalize embedding kullanır. Geçerli çalışma zamanı eşiği, hedef veri kalibrasyonu üretim değerini destekleyene kadar başlangıç ayarıdır.',
        'Reset demo': 'Demoyu sıfırla',
        'Clear courses, roster, enrollments, attempts, reviews, and audit history for a repeat presentation.': 'Tekrar sunum için dersleri, listeyi, kayıtları, denemeleri, incelemeleri ve denetim geçmişini temizler.',
        'Reset Demo': 'Demoyu Sıfırla',
        'No roster rows for this exam.': 'Bu sınav için liste kaydı yok.',
        'No roster rows match these filters.': 'Bu filtrelerle eşleşen liste kaydı yok.',
        'Attempt loaded. Review the images and status before taking action.': 'Deneme yüklendi. İşlem yapmadan önce görüntüleri ve durumu inceleyin.',
        'Loading attempt...': 'Deneme yükleniyor...',
        'Loading attempt details and previews.': 'Deneme ayrıntıları ve önizlemeler yükleniyor.',
        'Attempt could not be loaded': 'Deneme yüklenemedi',
        'Attempt could not be loaded.': 'Deneme yüklenemedi.',
        'No earlier attempts for this exam.': 'Bu sınav için önceki deneme yok.',
        'Select a student and at least one enrollment image.': 'Bir öğrenci ve en az bir kayıt görüntüsü seçin.',
        'enrollment for this exam': 'bu sınav için kayıt',
        'Select a student to view enrollment guidance.': 'Kayıt yönlendirmesini görmek için öğrenci seçin.',
        'Enrollment ready. Continue to exam-day selfie.': 'Kayıt hazır. Sınav günü selfiesine devam edin.',
        'This model needs a separate enrollment before verification.': 'Bu model doğrulamadan önce ayrı bir kayıt gerektirir.',
        'Add 3 clear samples before submitting the exam-day selfie.': 'Sınav günü selfiesini göndermeden önce 3 net örnek ekleyin.',
        'Enrollment is ready. Submit the exam-day selfie to request exam access.': 'Kayıt hazır. Sınav erişimi istemek için sınav günü selfiesini gönderin.',
        'Select an exam and student first.': 'Önce sınav ve öğrenci seçin.',
        'Select an exam, student, and exam-day selfie.': 'Sınav, öğrenci ve sınav günü selfiesi seçin.',
        'Verification cancelled. No attempt was recorded.': 'Doğrulama iptal edildi. Deneme kaydedilmedi.',
        'Preloaded demo selfie is staged. Press Verify for Exam Access to submit after confirmation.': 'Hazır demo selfiesi hazırlandı. Onaydan sonra göndermek için Sınav Erişimi İçin Doğrula düğmesine basın.',
        'Preloaded demo selfie staged. Confirm verification to submit.': 'Hazır demo selfiesi hazırlandı. Göndermek için doğrulamayı onaylayın.',
        'This student does not have a preloaded FLUXSynID selfie.': 'Bu öğrencinin hazır FLUXSynID selfiesi yok.',
        'Course saved.': 'Ders kaydedildi.',
        'Exam saved.': 'Sınav kaydedildi.',
        'Choose a CSV roster file.': 'Bir CSV liste dosyası seçin.',
        'Roster import complete.': 'Liste içe aktarma tamamlandı.',
        'FLUXSynID dataset not found on this machine.': 'Bu makinede FLUXSynID veri seti bulunamadı.',
        'Preuploading...': 'Ön yükleniyor...',
        'Exporting...': 'Dışa aktarılıyor...',
        'FLUXSynID preupload complete.': 'FLUXSynID ön yükleme tamamlandı.',
        'FLUXSynID test selfies exported.': 'FLUXSynID test selfieleri dışa aktarıldı.',
        'Demo reset.': 'Demo sıfırlandı.',
        'Open an attempt before reviewing.': 'İncelemeden önce bir deneme açın.',
        'Review saved.': 'İnceleme kaydedildi.',
        'No manual-review attempt exists yet. Submit a low-confidence or wrong-face selfie to create one.': 'Henüz manuel inceleme denemesi yok. Oluşturmak için düşük güvenli veya yanlış yüz selfiesi gönderin.',
        'Every rostered student is currently enrolled. Reset the demo or import a new roster row to show this path.': 'Listedeki tüm öğrenciler şu anda kayıtlı. Bu yolu göstermek için demoyu sıfırlayın veya yeni bir liste satırı içe aktarın.',
        'No model-mismatch enrollment exists for this exam yet. Change the exam model after enrolling a student to show this path.': 'Bu sınav için henüz model uyumsuzluğu kaydı yok. Bu yolu göstermek için öğrenciyi kaydettikten sonra sınav modelini değiştirin.',
        'No fallback request exists yet. Use Request Fallback after opening a review attempt.': 'Henüz yedek kontrol isteği yok. Bir inceleme denemesi açtıktan sonra Yedek Kontrol İste seçeneğini kullanın.',
        'No students in this exam roster.': 'Bu sınav listesinde öğrenci yok.',
        'No campus events yet.': 'Henüz kampüs olayı yok.',
        'No verification attempt selected yet.': 'Henüz doğrulama denemesi seçilmedi.',
        'Attempt Started': 'Deneme Başladı',
        'Photo Captured': 'Fotoğraf Çekildi',
        'Liveness Check': 'Canlılık Kontrolü',
        'Face Analyzed': 'Yüz Analiz Edildi',
        'Warnings present': 'Uyarı var',
        'Passed': 'Geçti',
        'Preloaded demo selfie': 'Hazır demo selfiesi',
        'Uploaded selfie': 'Yüklenen selfie',
        'Awaiting threshold': 'Eşik bekleniyor',
        'At or above threshold': 'Eşikte veya üzerinde',
        'Below threshold': 'Eşiğin altında',
        'Access can proceed': 'Erişim devam edebilir',
        'Auto-flagged': 'Otomatik işaretlendi',
        'Access denied': 'Erişim reddedildi',
        'Awaiting action': 'İşlem bekliyor',
        'Select a student to start the split-screen simulation.': 'Bölünmüş ekran simülasyonunu başlatmak için öğrenci seçin.',
        'Enrollment is missing. The student cannot verify until face samples are enrolled.': 'Kayıt eksik. Yüz örnekleri kaydedilene kadar öğrenci doğrulama yapamaz.',
        'Consent is required before the exam-day selfie can be submitted.': 'Sınav günü selfiesi gönderilmeden önce rıza gereklidir.',
        'Enrollment is ready. Submit an exam-day selfie to see the instructor dashboard update.': 'Kayıt hazır. Öğretim görevlisi panelinin güncellenmesini görmek için sınav günü selfiesi gönderin.',
        'Consent is required before camera or upload verification.': 'Kamera veya yükleme doğrulamasından önce rıza gereklidir.',
        'Submit or select an attempt before reviewing.': 'İncelemeden önce bir deneme gönderin veya seçin.',
        'Simulation review saved.': 'Simülasyon incelemesi kaydedildi.',
        'Choose two face images for model comparison.': 'Model karşılaştırması için iki yüz görüntüsü seçin.',
        'Access granted': 'Erişim verildi',
        'Proctor action needed': 'Gözetmen işlemi gerekli',
        'Exam access blocked': 'Sınav erişimi engellendi',
        'Enroll before verification': 'Doğrulamadan önce kayıt yapın',
        'Waiting for student': 'Öğrenci bekleniyor',
        'Verification in progress': 'Doğrulama sürüyor',
        'View': 'Görüntüle',
        'Open': 'Aç',
        'Review': 'İncele',
        'None': 'Yok',
        'Pass': 'Geçti',
        'Likely Match': 'Muhtemel Eşleşme',
        'Likely Different': 'Muhtemelen Farklı',
        'Model verified': 'Model doğruladı',
        'Manual review required': 'Manuel inceleme gerekli',
        'Access blocked': 'Erişim engellendi',
        'Model verified. Access can proceed if the exam policy allows it.': 'Model doğruladı. Sınav politikası izin veriyorsa erişim devam edebilir.',
        'Use proctor review or manual ID fallback before granting access.': 'Erişim vermeden önce gözetmen incelemesi veya manuel kimlik yedeğini kullanın.',
        'A proctor approved this access decision.': 'Bir gözetmen bu erişim kararını onayladı.',
        'A proctor has requested a fallback ID check before exam access.': 'Bir gözetmen sınav erişiminden önce yedek kimlik kontrolü istedi.',
        'A proctor must review this attempt before access is granted.': 'Erişim verilmeden önce bir gözetmen bu denemeyi incelemelidir.',
        'This attempt did not meet the exam access policy.': 'Bu deneme sınav erişim politikasını karşılamadı.',
    };

    const ATTRIBUTE_TEXT = {
        ...STATIC_TEXT,
        'Aylin': 'Aylin',
        'AT-2026': 'AT-2026',
    };

    const VALUE_TEXT = {
        'Manual ID check completed.': 'Manuel kimlik kontrolü tamamlandı.',
        'Midterm Exam': 'Ara Sınav',
        'Spring 2026': 'Bahar 2026',
        'SE 204 - Data Structures': 'SE 204 - Veri Yapıları',
    };

    const PATTERNS = [
        [/^Step (\d+) of 4$/, (_, step) => t('gate.step', { step })],
        [/^(.+) access gate$/i, (_, exam) => `${translateText(exam)} erişim kapısı`],
        [/^(.+) operations cockpit$/i, (_, course) => `${course} operasyon paneli`],
        [/^Threshold ([\d.-]+)$/, (_, threshold) => `Eşik ${threshold}`],
        [/^Threshold ([\d.-]+)\. Window (.+)\. Instructor (.+)\.$/, (_, threshold, windowText, instructor) => `Eşik ${threshold}. Zaman aralığı ${windowText}. Öğretim görevlisi ${instructor}.`],
        [/^(.+) verification before exam access$/, (_, model) => `Sınav erişiminden önce ${model} doğrulaması`],
        [/^(\d+) need review \| (\d+) verified \| (\d+) waiting \| (\d+) blocked$/, (_, review, verified, waiting, blocked) => t('queue.summary', { review, verified, waiting, blocked })],
        [/^(\d+) file\(s\) selected\. (\d+) enrollment slot\(s\) left\.$/, (_, files, slots) => `${files} dosya seçildi. ${slots} kayıt hakkı kaldı.`],
        [/^You selected (\d+) file\(s\), but only (\d+) enrollment slot\(s\) remain\.$/, (_, files, slots) => `${files} dosya seçtiniz, ancak yalnızca ${slots} kayıt hakkı kaldı.`],
        [/^Add (\d+) Sample(s?)$/, (_, count) => `${count} Örnek Ekle`],
        [/^Enrolled (\d+) sample\(s\)\.$/, (_, count) => `${count} örnek kaydedildi.`],
        [/^Imported (\d+); rejected (\d+)\.$/, (_, imported, rejected) => `${imported} içe aktarıldı; ${rejected} reddedildi.`],
        [/^Preuploaded (\d+) student\(s\); exported (\d+) test selfie file\(s\); skipped (\d+)\.$/, (_, imported, exported, skipped) => `${imported} öğrenci ön yüklendi; ${exported} test selfie dosyası dışa aktarıldı; ${skipped} atlandı.`],
        [/^Exported (\d+) matching selfie file\(s\); skipped (\d+)\.$/, (_, exported, skipped) => `${exported} eşleşen selfie dosyası dışa aktarıldı; ${skipped} atlandı.`],
        [/^(\d+) eligible identities found \| (\d+) students preuploaded$/, (_, eligible, preuploaded) => `${eligible} uygun kimlik bulundu | ${preuploaded} öğrenci ön yüklendi`],
        [/^(\d+) matching self(ie|ies) exported$/, (_, count) => `${count} eşleşen selfie dışa aktarıldı`],
        [/^Demo reset with (\d+) FLUXSynID faces\.$/, (_, count) => `Demo ${count} FLUXSynID yüzüyle sıfırlandı.`],
        [/^Demo reset\. FLUXSynID preupload skipped: (.+)$/, (_, error) => `Demo sıfırlandı. FLUXSynID ön yükleme atlandı: ${apiErrorMessage(error)}`],
        [/^(.+) - (Verified|Manual Review|Rejected|Approved by Proctor|Fallback Requested|Enrollment Needed|Not Started|Pending Verification)$/, (_, left, status) => `${left} - ${statusLabel(status)}`],
        [/^(.+) \| score ([\d.-]+) \| (.+)$/, (_, status, score, time) => `${statusLabel(status)} | skor ${score} | ${formatDateTime(time)}`],
        [/^(\d+) earlier attempt\(s\) for this exam\.$/, (_, count) => `Bu sınav için ${count} önceki deneme var.`],
        [/^Preloaded synthetic selfie staged for (.+)\. Press Verify for Exam Access to submit\.$/, (_, name) => `${name} için hazır sentetik selfie hazırlandı. Göndermek için Sınav Erişimi İçin Doğrula düğmesine basın.`],
        [/^Enrollment works, but add (\d+) more sample(s?) for a stronger prototype\.$/, (_, count) => `Kayıt çalışır durumda, ancak daha güçlü bir prototip için ${count} örnek daha ekleyin.`],
        [/^Add (\d+) more sample\(s\), or continue with lower reliability for demo purposes\.$/, (_, count) => `${count} örnek daha ekleyin veya demo amacıyla daha düşük güvenilirlikle devam edin.`],
        [/^This exam uses (.+); re-enroll for this model\.$/, (_, model) => `Bu sınav ${model} kullanıyor; bu model için yeniden kayıt yapın.`],
        [/^Exam now uses (.+)\. Students enrolled under (.+) must re-enroll\.$/, (_, current, previous) => `Sınav artık ${current} kullanıyor. ${previous} ile kayıtlı öğrenciler yeniden kaydolmalıdır.`],
        [/^Ready with (\d+) sample\(s\)$/, (_, count) => `${count} örnekle hazır`],
        [/^(\d+)\/3 samples; usable, but not ideal$/, (_, count) => `${count}/3 örnek; kullanılabilir ama ideal değil`],
        [/^Match Score: (.+)$/, (_, score) => `Eşleşme Skoru: ${score}`],
        [/^(.+) trend$/, (_, label) => `${translateText(label)} eğilimi`],
        [/^Course (.+) saved\.$/, (_, id) => `${id} dersi kaydedildi.`],
        [/^Exam (.+) saved\.$/, (_, id) => `${id} sınavı kaydedildi.`],
        [/^Imported (\d+) student\(s\) into (.+)\.$/, (_, count, course) => `${count} öğrenci ${course} dersine aktarıldı.`],
        [/^(.+) enrolled with (\d+) face sample\(s\)\.$/, (_, student, count) => `${student}, ${count} yüz örneğiyle kaydedildi.`],
        [/^(.+) preuploaded from FLUXSynID identity (.+)\.$/, (_, student, identity) => `${student}, FLUXSynID kimliği ${identity} üzerinden ön yüklendi.`],
        [/^(.+) produced (.+) for (.+)\.$/, (_, student, status, exam) => `${student}, ${exam} için ${statusLabel(status)} sonucu üretti.`],
        [/^(.+) applied to attempt (.+)\.$/, (_, action, attempt) => `${reviewActionLabel(action)} işlemi ${attempt} denemesine uygulandı.`],
    ];

    const API_PATTERNS = [
        [/^Student must complete face enrollment before exam verification\.$/, () => 'Öğrenci sınav doğrulamasından önce yüz kaydını tamamlamalı.'],
        [/^Student does not have a preuploaded FLUXSynID selfie\.$/, () => 'Öğrencinin hazır FLUXSynID selfiesi yok.'],
        [/^Student does not have a preuploaded FLUXSynID test selfie\.$/, () => 'Öğrencinin hazır FLUXSynID test selfiesi yok.'],
        [/^Preloaded demo selfie verification requires explicit confirmation\.$/, () => 'Hazır demo selfiesi doğrulaması açık onay gerektirir.'],
        [/^Select an exam and student first\.$/, () => 'Önce sınav ve öğrenci seçin.'],
        [/^Student is enrolled with (.+); this exam requires (.+)\.$/, (_, enrolled, required) => `Öğrenci ${enrolled} ile kayıtlı; bu sınav ${required} gerektiriyor.`],
        [/^Unknown student '(.+)'\.$/, (_, id) => `Bilinmeyen öğrenci: ${id}.`],
        [/^Unknown exam '(.+)'\.$/, (_, id) => `Bilinmeyen sınav: ${id}.`],
        [/^Student '(.+)' is not on this exam roster\.$/, (_, id) => `${id} bu sınav listesinde değil.`],
        [/^At least one image required\.$/, () => 'En az bir görüntü gerekli.'],
        [/^Upload at most 5 face samples\.$/, () => 'En fazla 5 yüz örneği yükleyin.'],
        [/^Empty file uploaded\.$/, () => 'Boş dosya yüklendi.'],
        [/^File too large \(max 10MB\)\.$/, () => 'Dosya çok büyük (en fazla 10 MB).'],
        [/^Roster must be UTF-8 CSV\.$/, () => 'Liste UTF-8 CSV olmalıdır.'],
        [/^CSV must include student_id,name,email columns\.$/, () => 'CSV student_id,name,email sütunlarını içermelidir.'],
        [/^Failed to (.+)$/, (_, action) => `${action} işlemi başarısız oldu.`],
    ];

    function interpolate(template, params = {}) {
        return String(template).replace(/\{(\w+)\}/g, (_, key) => params[key] ?? '');
    }

    function t(key, params = {}) {
        return interpolate(MESSAGES[key] || key, params);
    }

    function statusLabel(value) {
        return STATUS_LABELS[value] || value || '-';
    }

    function decisionLabel(value) {
        return DECISION_LABELS[value] || value || '-';
    }

    function eventLabel(value) {
        return EVENT_LABELS[value] || value || '-';
    }

    function reviewActionLabel(value) {
        const labels = {
            approve: 'onay',
            deny: 'ret',
            fallback: 'yedek kontrol',
        };
        return labels[value] || value;
    }

    function applyPatterns(text, patterns) {
        for (const [regex, replacer] of patterns) {
            const match = text.match(regex);
            if (match) return replacer(...match);
        }
        return null;
    }

    function apiErrorMessage(error) {
        const raw = typeof error === 'string' ? error : (error?.message || String(error || ''));
        const exact = STATIC_TEXT[raw] || STATUS_LABELS[raw] || DECISION_LABELS[raw];
        if (exact) return exact;
        return applyPatterns(raw, API_PATTERNS) || applyPatterns(raw, PATTERNS) || raw;
    }

    function translateText(value) {
        if (value === null || value === undefined) return value;
        const raw = String(value);
        if (!raw.trim()) return raw;
        const leading = raw.match(/^\s*/)?.[0] || '';
        const trailing = raw.match(/\s*$/)?.[0] || '';
        const text = raw.trim();
        const translated = STATIC_TEXT[text]
            || STATUS_LABELS[text]
            || DECISION_LABELS[text]
            || EVENT_LABELS[text]
            || applyPatterns(text, PATTERNS)
            || text;
        return `${leading}${translated}${trailing}`;
    }

    function formatDateTime(value, options = {}) {
        if (!value) return '-';
        const parsed = new Date(value);
        if (Number.isNaN(parsed.getTime())) return String(value).replace('T', ' ');
        const defaults = {
            dateStyle: 'medium',
            timeStyle: 'short',
        };
        return new Intl.DateTimeFormat(LOCALE, { ...defaults, ...options }).format(parsed);
    }

    function formatTime(value, options = {}) {
        const date = value instanceof Date ? value : new Date(value);
        if (Number.isNaN(date.getTime())) return String(value || '-');
        return date.toLocaleTimeString(LOCALE, {
            hour: '2-digit',
            minute: '2-digit',
            ...options,
        });
    }

    function translateAttributes(element) {
        ['aria-label', 'placeholder', 'title', 'data-preview'].forEach(attribute => {
            if (!element.hasAttribute(attribute)) return;
            const current = element.getAttribute(attribute);
            const translated = ATTRIBUTE_TEXT[current] || translateText(current);
            if (translated !== current) element.setAttribute(attribute, translated);
        });

        if (element.hasAttribute('data-i18n')) {
            element.textContent = t(element.getAttribute('data-i18n'));
        }
        if (element.hasAttribute('data-i18n-placeholder')) {
            element.setAttribute('placeholder', t(element.getAttribute('data-i18n-placeholder')));
        }
        if (element.hasAttribute('data-i18n-aria-label')) {
            element.setAttribute('aria-label', t(element.getAttribute('data-i18n-aria-label')));
        }
        if (element.hasAttribute('data-i18n-value')) {
            element.value = t(element.getAttribute('data-i18n-value'));
        } else if (['INPUT', 'TEXTAREA'].includes(element.tagName) && VALUE_TEXT[element.value]) {
            element.value = VALUE_TEXT[element.value];
        }
    }

    function translateTextNodes(root) {
        const walker = document.createTreeWalker(
            root,
            NodeFilter.SHOW_TEXT,
            {
                acceptNode(node) {
                    const parent = node.parentElement;
                    if (!parent || ['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(parent.tagName)) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    return node.nodeValue.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
                },
            }
        );
        const nodes = [];
        while (walker.nextNode()) nodes.push(walker.currentNode);
        nodes.forEach(node => {
            const translated = translateText(node.nodeValue);
            if (translated !== node.nodeValue) node.nodeValue = translated;
        });
    }

    let observer = null;
    let scheduled = false;

    function translateDom(root = document.body || document.documentElement) {
        if (!root) return;
        if (observer) observer.disconnect();

        document.documentElement.lang = 'tr';
        document.title = t('meta.title');
        const description = document.querySelector('meta[name="description"]');
        if (description) description.setAttribute('content', t('meta.description'));

        const elements = root.nodeType === Node.ELEMENT_NODE
            ? [root, ...root.querySelectorAll('*')]
            : [...document.querySelectorAll('*')];
        elements.forEach(translateAttributes);
        translateTextNodes(root.nodeType === Node.ELEMENT_NODE ? root : document.body);

        if (observer && document.body) {
            observer.observe(document.body, {
                subtree: true,
                childList: true,
                characterData: true,
                attributes: true,
                attributeFilter: ['aria-label', 'placeholder', 'title', 'data-preview'],
            });
        }
    }

    function scheduleTranslation() {
        if (scheduled) return;
        scheduled = true;
        window.requestAnimationFrame(() => {
            scheduled = false;
            translateDom();
        });
    }

    function startObserver() {
        if (!('MutationObserver' in window) || !document.body) return;
        observer = new MutationObserver(scheduleTranslation);
        observer.observe(document.body, {
            subtree: true,
            childList: true,
            characterData: true,
            attributes: true,
            attributeFilter: ['aria-label', 'placeholder', 'title', 'data-preview'],
        });
    }

    document.addEventListener('DOMContentLoaded', () => {
        translateDom();
        startObserver();
    });

    window.FaceVerifyI18n = {
        locale: LOCALE,
        t,
        statusLabel,
        decisionLabel,
        eventLabel,
        reviewActionLabel,
        formatDateTime,
        formatTime,
        apiErrorMessage,
        translateText,
        translateDom,
    };
    window.t = t;
    window.statusLabel = statusLabel;
    window.decisionLabel = decisionLabel;
    window.eventLabel = eventLabel;
    window.reviewActionLabel = reviewActionLabel;
    window.formatDateTime = formatDateTime;
    window.formatTime = formatTime;
    window.apiErrorMessage = apiErrorMessage;
    window.localizeText = translateText;
    window.localizeDom = translateDom;
})();
