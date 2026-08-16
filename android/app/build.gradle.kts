import java.io.File
import java.util.Properties

plugins {
    id("com.android.application")
}

val keystorePropertiesFile = project.findProperty("releaseKeystoreProperties")?.let {
    file(it.toString())
} ?: rootProject.file("keystore.properties")
val keystoreProperties = Properties().apply {
    if (keystorePropertiesFile.exists()) {
        keystorePropertiesFile.inputStream().use { load(it) }
    }
}

android {
    namespace = "com.shiyu.hccr"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.shiyu.hccr"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        debug {
            externalNativeBuild {
                cmake {
                    arguments("-DCMAKE_BUILD_TYPE=Release")
                }
            }
        }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            val storeFilePath = keystoreProperties.getProperty("storeFile")
            if (storeFilePath != null) {
                signingConfig = signingConfigs.create("release") {
                    val configuredFile = File(storeFilePath)
                    storeFile = if (configuredFile.isAbsolute) configuredFile
                                else File(keystorePropertiesFile.parentFile, storeFilePath)
                    storePassword = keystoreProperties.getProperty("storePassword")
                    keyAlias = keystoreProperties.getProperty("keyAlias")
                    keyPassword = keystoreProperties.getProperty("keyPassword")
                }
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/jni/CMakeLists.txt")
            version = "3.22.1+"
        }
    }

    ndkVersion = "30.0.14904198"
}

dependencies {
    implementation("androidx.core:core:1.15.0")
}
